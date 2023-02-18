import copy, itertools
import os
import sys
import math
import glob
import uuid
import shutil
import random
import tempfile
import importlib
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict

from huggingface_hub import HfApi, HfFolder, Repository

SAMPLE_RATE = 16000

MODEL_CARD_MARKDOWN = """---
datasets:
- superb
tags:
- library:s3prl
- benchmark:superb
- type:model
---

# Fine-tuned s3prl model

Upstream Model: {upstream_model}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""

def fix_seed(args):
    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_random_state():
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    random_state = random.getstate()
    return torch_rng_state, np_rng_state, random_state

def set_random_state(state):
    torch_rng_state, np_rng_state, random_state = state
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_state)

def cal_params_diff(model1, model2):
    ddc = 0
    ttc = 0
    for (n1, p1), (n2,p2) in zip(model1.model.named_parameters(), model2.model.named_parameters()):
        ddc += torch.sum((p1 != p2).float()).item()
        ttc += p1.numel()
    print(ddc / ttc)
    return ddc / ttc

class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        
        if args.mode == 'train':
            self.upstream, self.featurizer, self.downstream,  self.all_entries = None, None, None, None
            self.upstream = self._get_upstream()

            self.upstream_1 = self._get_upstream()

            self.featurizer_1 = self._get_featurizer(self.upstream_1)
            self.downstream_1 = self._get_downstream(self.featurizer_1)
            self.all_entries_1 = [self.upstream_1, self.featurizer_1, self.downstream_1]
            
            #deepcopy upstream模型forward为空，不知道为什么

            self.upstream_2 = self._get_upstream()
            self.featurizer_2 = self._get_featurizer(self.upstream_2)
            self.downstream_2 = self._get_downstream(self.featurizer_2)
            self.all_entries_2 = [self.upstream_2, self.featurizer_2, self.downstream_2]

            self.upstream_3 = self._get_upstream()
            self.featurizer_3 = self._get_featurizer(self.upstream_3)
            self.downstream_3 = self._get_downstream(self.featurizer_3)
            self.all_entries_3 = [self.upstream_3, self.featurizer_3, self.downstream_3]

            # self.upstream_4 = self._get_upstream()
            # self.featurizer_4 = self._get_featurizer(self.upstream_4)
            # self.downstream_4 = self._get_downstream(self.featurizer_4)
            # self.all_entries_4 = [self.upstream_4, self.featurizer_4, self.downstream_4]

            self.upstreams = [self.upstream_1, self.upstream_2, self.upstream_3] #
            self.featurizers = [self.featurizer_1, self.featurizer_2, self.featurizer_3] # 
            self.downstreams = [self.downstream_1, self.downstream_2, self.downstream_3] # 

            self.all_subnets_all_entries = [self.all_entries_1, self.all_entries_2, self.all_entries_3] #

        else:
            self.upstream = self._get_upstream()
            self.featurizer = self._get_featurizer(self.upstream)
            self.downstream = self._get_downstream(self.featurizer)
            self.all_entries = [self.upstream, self.featurizer, self.downstream]

        self.dev_score = []
        self.ob_mode = self.config['runner'].get('observation', ['train', 'loss'])[0]
        self.ob_target = self.config['runner'].get('observation', ['train', 'loss'])[1]

        print('-------------------', self.ob_mode, self.ob_target, '-------------------')

    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)


    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)


    def _get_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            print(f'[Runner] - Downloading upstream model {self.args.upstream} from the Hugging Face Hub')
            filepath = snapshot_download(self.args.upstream, self.args.upstream_revision, use_auth_token=True)
            sys.path.append(filepath)

            dependencies = (Path(filepath) / 'requirements.txt').resolve()
            print("[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
            for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
                print(f"{idx}. {line.strip()}")
            print(f"You can install them by:")
            print()
            print(f"pip install -r {dependencies}")
            print()

            from expert import UpstreamExpert
            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt

        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable,
            interfaces = ["get_downsample_rates"]
        )


    def _get_featurizer(self, upstream):
        model = Featurizer(
            upstream = upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )


    def _get_downstream(self, featurizer):
        expert = importlib.import_module(f"s3prl.downstream.{self.args.downstream}.expert")
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            upstream_dim = featurizer.model.output_dim,
            upstream_rate = featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args)
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Downstream',
            trainable = True,
            interfaces = ['get_dataloader', 'log_records']
        )


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )
        self._load_weight(optimizer, 'Optimizer')
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )
        self._load_weight(scheduler, 'Scheduler')
        return scheduler

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(upstream_model=self.args.upstream)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)

    def get_fisher_mask(self):
        grad_masks = [dict() for i in range(len(self.all_subnets_all_entries))]
        #self.upstream.model.train()
        
        records = defaultdict(list)

        for entry in self.all_subnets_all_entries[0]:
            entry.model.train()
        # for entry in self.all_entries:
        #      entry.model.train()
        tuning_pcount = 0
        pcount = 0
        # specaug
        specaug = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])

        for idx, (named_parameters, grad_mask) in enumerate(zip((up.model.named_parameters() for up in self.upstreams), grad_masks)):
            print(idx, named_parameters, grad_mask)
            for name, params in named_parameters:
                if 'encoder.layers' in name or 'layer_norm' in name:
                    if idx == 0:
                        print(f"encoder.layers or layer_norm : {name}")
                        tuning_pcount += params.numel()
                    grad_mask[params] = params.new_zeros(params.size())
                else:
                    print('frozen: ', name)      
                    params.requires_grad = False   
                if 'final_proj'  not in name:
                    if idx == 0:
                        pcount += params.numel()

        # for (name_1, params_1), (name_2, params_2), (name_3, params_3) in zip(self.upstreams[0].model.named_parameters(), self.upstreams[1].model.named_parameters(), self.upstreams[2].model.named_parameters()):    #bug !!! dif model have diff param

        #     if 'encoder.layers' in name_1 or 'layer_norm' in name_1:
        #         print("encoder.layers or layer_norm : ", name_1)
        #         grad_masks[0][params_1] = params_1.new_zeros(params_1.size())
        #         grad_masks[1][params_2] = params_1.new_zeros(params_2.size())
        #         grad_masks[2][params_3] = params_1.new_zeros(params_3.size())
        #         tuning_pcount += params_1.numel()
        #     else:
        #         print('frozen: ', name_1)      
        #         params_1.requires_grad = False

        #     if 'final_proj'  not in name_1:
        #         pcount += params_1.numel()
        
        tuning_pcount *= np.array(self.config['optimizer']['reserve_p'])

        print(f'num of tuning params: {tuning_pcount / 1e6} M')
        print(f'num of total params: {pcount / 1e6} M')
        
        # Now begin
        train_split = self.config['runner'].get("train_dataloader", "train")
        train_dataloader = self.downstreams[0].model.get_dataloader(train_split)
        
        N = len(train_dataloader)

        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')   #如果 is_leader_process() 返回 True，则进度条将显示在命令行中，否则，进度条将被忽略。
        for batch_id, (wavs, *others) in enumerate(tqdm(train_dataloader, dynamic_ncols=True, desc='get_fisher', file=tqdm_file)):
            try:
                wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]

                features = self.upstreams[0].model(wavs)
                features = self.featurizers[0].model(wavs, features)

                if specaug:
                    features, _ = specaug(features)

                loss = self.downstreams[0].model(
                    train_split,
                    features, *others,
                    records = records,
                )

                gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                (loss / gradient_accumulate_steps).backward()

                for (name_1, params_1), (name_2, params_2), (name_3, params_3) in zip(self.upstreams[0].model.named_parameters(), self.upstreams[1].model.named_parameters(), self.upstreams[2].model.named_parameters()): 
                    if 'encoder.layers' in name_1 or 'layer_norm' in name_1: #应该laynorn也，不过影响不大
                        torch.nn.utils.clip_grad_norm_(params_1, self.config['runner']['gradient_clipping'])
                        fisher_item = (params_1.grad ** 2) / N 
                        grad_masks[0][params_1] += fisher_item
                        grad_masks[1][params_2] = grad_masks[0][params_1]
                        grad_masks[2][params_3] = grad_masks[0][params_1]
                        #print(type(grad_masks[0][params_1]))
                        # for grad_mask in grad_masks:
                        #     grad_mask[params] += fisher_item              #累计梯度
                
                self.upstreams[0].model.zero_grad()         
                for entry in self.all_subnets_all_entries[0]:
                    entry.model.zero_grad()
                del loss
            except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {batch_id}')
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        for entry in self.all_subnets_all_entries[0]:
                            entry.model.zero_grad()
                        continue
                    else:
                        raise
        print('Calculate Fisher Information')

        # Numpy
        r = None
        for k, v in grad_masks[2].items():
            v = v.view(-1).cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)
        
        polars = [np.percentile(r, (1-reserve_p)*100) for reserve_p in self.config['optimizer']['reserve_p']]
        for i, grad_mask in enumerate(grad_masks):
            for k in grad_mask:
                grad_mask[k] = grad_mask[k] >= polars[i]
        print('candidate P => {}; Polar => {}'.format(self.config['optimizer']['reserve_p'], polars))
        # sum = [0, 0, 0]
        # for i, grad_mask in enumerate(grad_masks):   
        #     for k in grad_mask:
        #         sum[i] += (grad_mask[k] == True).sum()
        # print(sum)
        # exit()             #折没问题
        # TODO: pytorch: torch.kthvalue
        
        return grad_masks

    def train(self):

        assert len(self.config['optimizer']['reserve_p']) == len(self.all_subnets_all_entries) , "Please check the num of candidate P and network is equal"
        # trainable parameters and train/eval mode
        trainable_models_list = []
        trainable_paras_list = []
        for all_entries in self.all_subnets_all_entries:
        #for all_entries in [self.all_entries]:
            trainable_models = []
            trainable_paras = []
            for entry in all_entries:
                if entry.trainable:
                    entry.model.train()
                    trainable_models.append(entry.model)
                    trainable_paras += list(entry.model.parameters())
                else:
                    entry.model.eval()
            trainable_models_list.append(trainable_models)
            trainable_paras_list.append(trainable_paras)

        optimizers = [self._get_optimizer(trainable_models) for trainable_models in trainable_models_list]

        # scheduler
        #scheduler = None
        schedulers = [None for i in range(len(self.all_subnets_all_entries))]
        if self.config.get('scheduler'):
            #scheduler = self._get_scheduler(optimizer)
            schedulers = [self._get_scheduler(optimizer) for optimizer in optimizers]

        # specaug
        specaug = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        #selete_pbars = [tqdm(total=self.config['runner']['total_steps'] * 0.1, dynamic_ncols=True, desc='overall', file=tqdm_file) for i in range(3)]
        pbar = tqdm(total=self.config['runner']['total_steps'], dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        epoch = self.init_ckpt.get('Epoch', 0)
        optim_p = None
        train_split = self.config['runner'].get("train_dataloader", "train")
        records = [defaultdict(list) for i in range(len(self.all_subnets_all_entries))]
        selete_p = self.init_ckpt.get('Optim_p', -1)
        min_idx = 0
        strive_ratio = self.config['runner'].get('strive_ratio', 0.1)
        strive_steps = self.config['runner']['total_steps'] * strive_ratio
        observation = 'loss'
        print(f"-------------------------strive_ratio : {strive_steps}  observation : {observation}----------------------")

        batch_ids = []
        backward_steps = 0
        # =================== HACK BEGIN =======================   
        #get_fisher_mask & set mask
        if self.args.tuning_mode == "subnet":
            print(f'[Runner] - Here is Subnet Tuning')
            grad_masks = self.get_fisher_mask()
            for i in range(len(optimizers)):
                optimizers[i].set_grad_mask(grad_masks[i])

            if self.upstreams[0].trainable:
                print("upstream is tuning")
            if selete_p != -1:  #resume
                min_idx = self.config['optimizer']['reserve_p'].index(selete_p)
                self.upstream = self.upstreams[min_idx]
                self.featurizer = self.featurizers[min_idx]
                self.downstream = self.downstreams[min_idx]
                self.all_entries = self.all_subnets_all_entries[min_idx]
                trainable_paras = trainable_paras_list[min_idx]
                optimizer = optimizers[min_idx]
                scheduler = schedulers[min_idx]
                records = records[min_idx]
                for i in range(len(self.all_subnets_all_entries)):
                    if i != min_idx:
                        exec(f"del self.upstream_{i+1}")
                        exec(f"del self.featurizer_{i+1}")
                        exec(f"del self.downstream_{i+1}")
                optim_p = self.config['optimizer']['reserve_p'][min_idx]
                print(f"selete p => {self.config['optimizer']['reserve_p'][min_idx]}")
        # =================== HACK END =========================  

        # =================== HACK BEGIN =======================  
        # model-seletion dependon on training loss
        # =================== HACK END =========================  


        # 定义滑动平均系数
        alpha = 0.6
        last = [0 for i in range(len(self.all_subnets_all_entries))]
        smoothed_value = [0 for i in range(len(self.all_subnets_all_entries))]
        num_acc = 0
        
        global_step = 0
        strive_bar = tqdm(total=strive_steps, dynamic_ncols=True, desc=f'strive', file=tqdm_file)
        #可视化
        log_losses = [[] for i in range(len(self.all_subnets_all_entries))]
        while pbar.n < pbar.total:
            try:
                dataloader = self.downstreams[0].model.get_dataloader(train_split, epoch=epoch)
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    dataloader = self.downstreams[0].model.get_dataloader(train_split)
                    if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
                        dataloader.sampler.set_epoch(epoch)
                else:
                    raise

            for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                # try/except block for forward/backward
                
                wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                batch_ids.append(batch_id)
                if pbar.n < strive_steps:
                    global_step = pbar.n + 1
                    for i in range(len(self.all_subnets_all_entries)):
                        try:
                            if self.upstreams[i].trainable:
                                features = self.upstreams[i].model(wavs)

                            else:
                                with torch.no_grad():
                                    features = self.upstreams[i].model(wavs)
 
                            features = self.featurizers[i].model(wavs, features)

                            if specaug:
                                features, _ = specaug(features)

                            loss = self.downstreams[i].model(
                                train_split,
                                features, *others,
                                records = records[i],
                            )
                            
                            gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                            (loss / gradient_accumulate_steps).backward()
                            del loss

                        except RuntimeError as e:
                            if 'CUDA out of memory' in str(e):
                                print(f'[Runner] - CUDA out of memory at step {global_step}')
                                if is_initialized():
                                    raise
                                with torch.cuda.device(self.args.device):
                                    torch.cuda.empty_cache()
                                optimizers[i].zero_grad()
                                continue
                            else:
                                raise

                        # whether to accumulate gradient
                        if i == 0:   #只累计一次
                            backward_steps += 1
                        if backward_steps % gradient_accumulate_steps > 0:
                            continue

                        # gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_paras_list[i], self.config['runner']['gradient_clipping'])

                        # optimize
                        if math.isnan(grad_norm):
                            print(f'[Runner] - grad norm is NaN at step {global_step}')
                        else:
                            optimizers[i].step()
                        optimizers[i].zero_grad()
                        # print('-------------------------------')
                        # cal_params_diff(self.upstreams[i], self.upstream)
                        # adjust learning rate
                        if schedulers[i]:
                            schedulers[i].step()

                        if not is_leader_process():
                            batch_ids = []
                            records = [defaultdict(list) for j in range(len(self.all_subnets_all_entries))]
                            continue
                        if i == 0:
                            pbar.update(1)
                            strive_bar.update(1)
                        
                        if global_step % self.config['runner']['eval_step'] == 0:
                            self.strive_evaluate(self.upstreams[i], self.featurizers[i], self.downstreams[i], self.all_subnets_all_entries[i], self.config['runner']['eval_dataloaders'][0], logger, global_step)

                    # logging
                    if global_step % self.config['runner']['log_step'] == 0:

                        # 对数据进行平滑
                        num_acc += 1
                        debias_weight = 1.0 if alpha == 1.0 else 1.0 - alpha ** num_acc
                        for i in range(len(self.all_subnets_all_entries)):
                            self.downstreams[i].model.log_records(
                            train_split,
                            records = records[i],
                            logger = logger,
                            global_step = global_step,
                            batch_ids = batch_ids,
                            total_batch_num = len(dataloader),
                            )
                            print("cur value", sum(records[i][observation]) / len(records[i][observation]))
                            cur_value = sum(records[i][observation]) / len(records[i][observation])
                            log_losses[i].append(cur_value)
                            last[i] = alpha * last[i] + (1 - alpha) * cur_value
                            smoothed_value[i] = last[i] / debias_weight
                        print("smoothed_value : ", smoothed_value)
                        batch_ids = []
                        records = [defaultdict(list) for j in range(len(self.all_subnets_all_entries))]                  


                    if pbar.n == strive_steps:    #选则最优的p
                        if self.ob_mode == 'train':  # train loss
                            min_idx = smoothed_value.index(min(smoothed_value))
                        elif self.ob_mode == 'dev' or self.ob_mode == 'per':   #dev 
                            can_val = [self.dev_score[-3], self.dev_score[-2], self.dev_score[-1]]
                            if self.ob_target == 'wer':
                                min_idx = can_val.index(min(can_val))
                            else:    #acc or f1
                                min_idx = can_val.index(max(can_val))
                            
                        self.upstream = self.upstreams[min_idx]
                        self.featurizer = self.featurizers[min_idx]
                        self.downstream = self.downstreams[min_idx]
                        self.all_entries = self.all_subnets_all_entries[min_idx]
                        trainable_paras = trainable_paras_list[min_idx]
                        optimizer = optimizers[min_idx]
                        scheduler = schedulers[min_idx]
                        records = records[min_idx]
                        
                        for i in range(len(self.all_subnets_all_entries)):
                            if i != min_idx:
                                exec(f"del self.upstream_{i+1}")
                                exec(f"del self.featurizer_{i+1}")
                                exec(f"del self.downstream_{i+1}")
                                loss_path = os.path.join(self.args.expdir, 'loss_' + str(self.config['optimizer']['reserve_p'][i]))
                                np.save(f'{loss_path}.npy', np.array(log_losses[i]))
                        optim_p = self.config['optimizer']['reserve_p'][min_idx]
                        print(f"selete p => {self.config['optimizer']['reserve_p'][min_idx]}")

                            
                else:
                    try:
                        if pbar.n >= pbar.total:
                                break
                        global_step = pbar.n + 1
                        if self.upstream.trainable:
                            features = self.upstream.model(wavs)
                        else:
                            with torch.no_grad():
                                features = self.upstream.model(wavs)
                        features = self.featurizer.model(wavs, features)

                        if specaug:
                            features, _ = specaug(features)

                        loss = self.downstream.model(
                            train_split,
                            features, *others,
                            records = records,
                        )
                        batch_ids.append(batch_id)

                        gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                        (loss / gradient_accumulate_steps).backward()
                        del loss

                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print(f'[Runner] - CUDA out of memory at step {global_step}')
                            if is_initialized():
                                raise
                            with torch.cuda.device(self.args.device):
                                torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            continue
                        else:
                            raise

                    # whether to accumulate gradient
                    backward_steps += 1
                    if backward_steps % gradient_accumulate_steps > 0:
                        continue

                    # gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_paras, self.config['runner']['gradient_clipping'])

                    # optimize
                    if math.isnan(grad_norm):
                        print(f'[Runner] - grad norm is NaN at step {global_step}')
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # adjust learning rate
                    if scheduler:
                        scheduler.step()

                    if not is_leader_process():
                        batch_ids = []
                        records = defaultdict(list)
                        continue

                    # logging
                    if global_step % self.config['runner']['log_step'] == 0:
                        self.downstream.model.log_records(
                            train_split,
                            records = records,
                            logger = logger,
                            global_step = global_step,
                            batch_ids = batch_ids,
                            total_batch_num = len(dataloader),
                        )
                        batch_ids = []
                        records = defaultdict(list)

                    # evaluation and save checkpoint
                    save_names = []

                    if global_step % self.config['runner']['eval_step'] == 0:
                        for split in self.config['runner']['eval_dataloaders']:
                            save_names += self.evaluate(split, logger, global_step)

                    if global_step % self.config['runner']['save_step'] == 0:
                        def check_ckpt_num(directory):
                            max_keep = self.config['runner']['max_keep']
                            ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                            if len(ckpt_pths) >= max_keep:
                                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                                for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                    os.remove(ckpt_pth)
                        check_ckpt_num(self.args.expdir)
                        save_names.append(f'states-{global_step}.ckpt')

                    if len(save_names) > 0:
                        all_states = {
                            'Optimizer': optimizer.state_dict(),
                            'Step': global_step,
                            'Epoch': epoch,
                            'Args': self.args,
                            'Config': self.config,
                            'Optim_p': optim_p
                        }

                        for entry in self.all_entries:
                            if entry.trainable:
                                all_states[entry.name] = get_model_state(entry.model)

                        if scheduler:
                            all_states['Scheduler'] = scheduler.state_dict()

                        if is_initialized():
                            all_states['WorldSize'] = get_world_size()

                        save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                        tqdm.write(f'[Runner] - Save the checkpoint to:')
                        for i, path in enumerate(save_paths):
                            tqdm.write(f'{i + 1}. {path}')
                            torch.save(all_states, path)

                    pbar.update(1)
            epoch += 1

        pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()


    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""
        if 'Optim_p' in self.init_ckpt:
            print('cur_p is ', self.init_ckpt['Optim_p'])
        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_entries:
            trainings.append(entry.model.training)
            entry.model.eval()

        # prepare data
        dataloader = self.downstream.model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
            if batch_id > evaluate_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            with torch.no_grad():
                features = self.upstream.model(wavs)
                features = self.featurizer.model(wavs, features)
                self.downstream.model(
                    split,
                    features, *others,
                    records = records,
                    batch_id = batch_id,
                )
                batch_ids.append(batch_id)

        save_names = self.downstream.model.log_records(
            split,
            records = records,
            logger = logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
        )

        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(self.all_entries, trainings):
            if training:
                entry.model.train()

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names

    def strive_evaluate(self, upstream, featurizer, downstream, all_entries, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""
        
        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in all_entries:
            trainings.append(entry.model.training)
            entry.model.eval()

        # prepare data
        dataloader = downstream.model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
            if batch_id > evaluate_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            with torch.no_grad():
                features = upstream.model(wavs)
                features = featurizer.model(wavs, features)
                downstream.model(
                    split,
                    features, *others,
                    records = records,
                    batch_id = batch_id,
                )
                batch_ids.append(batch_id)

        save_names = downstream.model.log_records(
            split,
            records = records,
            logger = logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
        )

        if self.ob_mode == 'dev':
            self.dev_score.append(torch.FloatTensor(records[self.ob_target]).mean().item())
            print('cur dev_score : ', self.dev_score)
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(all_entries, trainings):
            if training:
                entry.model.train()

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        
    
    def inference(self):
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval()

        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            self.downstream.model.inference(features, [filename])

    def push_to_huggingface_hub(self):
        """Creates a downstream repository on the Hub and pushes training artifacts to it."""
        if self.args.hf_hub_org.lower() != "none":
            organization = self.args.hf_hub_org
        else:
            organization = os.environ.get("HF_USERNAME")
        huggingface_token = HfFolder.get_token()
        print(f"[Runner] - Organisation to push fine-tuned model to: {organization}")
        
        # Extract upstream repository metadata
        if self.args.hub == "huggingface":
            model_info = HfApi().model_info(self.args.upstream, token=huggingface_token)
            downstream_model_id = model_info.sha
            # Exclude "/" characters from downstream repo ID
            upstream_model_id = model_info.modelId.replace("/", "__")
        else:
            upstream_model_id = self.args.upstream.replace("/", "__")
            downstream_model_id = str(uuid.uuid4())[:8]
        repo_name = f"{upstream_model_id}__{downstream_model_id}"
        # Create downstream repo on the Hub
        repo_url = HfApi().create_repo(
            token=huggingface_token,
            name=repo_name,
            organization=organization,
            exist_ok=True,
            private=False,
        )
        print(f"[Runner] - Created Hub repo: {repo_url}")

        # Download repo
        HF_HUB_DIR = "hf_hub"
        REPO_ROOT_DIR = os.path.join(self.args.expdir, HF_HUB_DIR, repo_name)
        REPO_TASK_DIR = os.path.join(REPO_ROOT_DIR, self.args.downstream, self.args.expname)
        print(f"[Runner] - Cloning Hub repo to {REPO_ROOT_DIR}")
        model_repo = Repository(
            local_dir=REPO_ROOT_DIR, clone_from=repo_url, use_auth_token=huggingface_token
        )
        # Pull latest changes if they exist
        model_repo.git_pull()

        # Copy checkpoints, tensorboard logs, and args / configs
        # Note that this copies all files from the experiment directory,
        # including those from multiple runs
        shutil.copytree(self.args.expdir, REPO_TASK_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns(HF_HUB_DIR))

        # By default we use model.ckpt in the PreTrainedModel interface, so
        # rename the best checkpoint to match this convention
        checkpoints = list(Path(REPO_TASK_DIR).glob("*best*.ckpt"))
        if len(checkpoints) == 0:
            print("[Runner] - Did not find a best checkpoint! Using the final checkpoint instead ...")
            CKPT_PATH = (
                os.path.join(REPO_TASK_DIR, f"states-{self.config['runner']['total_steps']}.ckpt")
                )
        elif len(checkpoints) > 1:
            print(f"[Runner] - More than one best checkpoint found! Using {checkpoints[0]} as default ...")
            CKPT_PATH = checkpoints[0]
        else:
            print(f"[Runner] - Found best checkpoint {checkpoints[0]}!")
            CKPT_PATH = checkpoints[0]
        shutil.move(CKPT_PATH, os.path.join(REPO_TASK_DIR, "model.ckpt"))
        model_repo.lfs_track("*.ckpt")

        # Write model card
        self._create_model_card(REPO_ROOT_DIR)

        # Push everything to the Hub
        print("[Runner] - Pushing model files to the Hub ...")
        model_repo.push_to_hub()
        print("[Runner] - Training run complete!")
