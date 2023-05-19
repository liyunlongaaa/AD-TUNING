# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the speaker diarization dataset ]
#   Source       [ Refactored from https://github.com/hitachi-speech/EEND ]
#   Author       [ Jiatong Shi ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os
import math
import h5py
import random
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

# -------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized, get_rank
from torch.nn.utils.rnn import pad_sequence

# -------------#
from .model import Model
from .dataset import DiarizationDataset
from .utils import pit_loss, calc_diarization_error, get_label_perm


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        self.datarc = downstream_expert["datarc"]

        config_frame_shift = self.datarc.get("frame_shift")
        if isinstance(config_frame_shift, int):
            logging.warning(
                f"Diarization label frame shfit: {config_frame_shift}. "
                "It is set in the config field. You don't need to set this config field if "
                "you are training new downstream models. This module will then automatically "
                "use upstream's downsample rate as the training label frame shift. This "
                "'if condition' is designed only to inference the already trained downstream "
                "checkpoints with the command: python3 run_downstream.py -m evaluate -e [ckpt]. "
                "The checkpoint contains the frame_shift used for its training, and the same "
                "frame_shift should be prepared for the inference."
            )
            frame_shift = config_frame_shift
        else:
            logging.warning(
                f"Diarization label frame shfit: {upstream_rate}. It is automatically set as "
                "upstream's downsample rate to best align the representation v.s. labels for training. "
                "This frame_shift information will be saved in the checkpoint for future inference."
            )
            frame_shift = upstream_rate

        self.datarc["frame_shift"] = frame_shift
        with (Path(expdir) / "frame_shift").open("w") as file:
            print(frame_shift, file=file)

        self.loaderrc = downstream_expert["loaderrc"]
        self.modelrc = downstream_expert["modelrc"]
        self.scorerc = downstream_expert["scorerc"]

        self.train_batch_size = self.loaderrc["train_batchsize"]
        self.eval_batch_size = self.loaderrc["eval_batchsize"]

        self.expdir = expdir
        self.score_dir = os.path.join(expdir, "scoring")
        self.save_predictions = self.scorerc["save_predictions"]

        if ((not is_initialized()) or get_rank() == 0) \
                and not os.path.exists(self.score_dir) and self.save_predictions:
            os.makedirs(os.path.join(self.score_dir, "predictions"))

        self.model = Model(
            input_dim=self.upstream_dim,
            output_class_num=self.datarc["num_speakers"],
            **self.modelrc,
        )
        self.objective = pit_loss

        self.logging = os.path.join(expdir, "log.log")
        self.register_buffer("best_score", torch.zeros(1))

    # Interface
    def get_dataloader(self, mode):
        """
        Args:
            mode: string
                'train', 'dev' or 'test'
        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:
            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...
            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """
        if not hasattr(self, f"{mode}_dataset"):
            dataset = DiarizationDataset(
                mode,
                self.loaderrc[f"{mode}_dir"],
                **self.datarc,
            )
            setattr(self, f"{mode}_dataset", dataset)

        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_dev_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_test_dataloader(self.test_dataset)

    """
    Datalaoder Specs:
        Each dataloader should output in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==16000
    """

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_dev_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_test_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn_rec_infer,
        )

    def _tile_representations(self, reps, factor):
        """
        Tile up the representations by `factor`.
        Input - sequence of representations, shape: (batch_size, seq_len, feature_dim)
        Output - sequence of tiled representations, shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(reps.shape) == 3
        ), "Input argument `reps` has invalid shape: {}".format(reps.shape)
        tiled_reps = reps.repeat(1, 1, factor)
        tiled_reps = tiled_reps.reshape(
            reps.size(0), reps.size(1) * factor, reps.size(2)
        )
        return tiled_reps

    def _match_length(self, inputs, labels):
        """
        Since the upstream extraction process can sometimes cause a mismatch
        between the seq lenth of inputs and labels:
        - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
        - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
        Note that the length of labels should never be changed.
        """
        input_len, label_len = inputs.size(1), labels.size(1)

        factor = int(round(label_len / input_len))
        if factor > 1:
            inputs = self._tile_representations(inputs, factor)
            input_len = inputs.size(1)

        if input_len > label_len:
            inputs = inputs[:, :label_len, :]
        elif input_len < label_len:
            pad_vec = inputs[:, -1, :].unsqueeze(1)  # (batch_size, 1, feature_dim)
            inputs = torch.cat(
                (inputs, pad_vec.repeat(1, label_len - input_len, 1)), dim=1
            )  # (batch_size, seq_len, feature_dim), where seq_len == labels.size(-1)
        return inputs, labels

    # Interface
    def forward(self, mode, features, labels, lengths, rec_id, records, **kwargs):
        """
        Args:
            mode: string
                'train', 'dev' or 'test' for this forward step

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            labels:
                the frame-wise speaker labels

            rec_id:
                related recording id, use for inference

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        #print("labels.shape",len(labels), labels[0].shape)
        #print("features.shape",len(features),features[0].shape)
        #print("lengths",lengths)    #lengths也是list，上面3个长都是batch
        labels = [torch.from_numpy(label) for label in labels]   #[（t,2）,（t,2）....]b个
        lengths = torch.LongTensor(lengths)

        features = pad_sequence(features, batch_first=True)   #(b,t,dim)
        #print("pad_sequence features.shape",features.shape)
        labels = pad_sequence(labels, batch_first=True, padding_value=0).to(
            features.device
        )
        #print("pad_sequence labels.shape",labels.shape)
        features, labels = self._match_length(features, labels)  #使得两个t一样
        #print("_match_length", features.shape, labels.shape)
        predicted = self.model(features)       #（b,t,2）
        #print("predicted predicted.shape",predicted.shape)
        # cause logits are in (batch, seq, class) and labels are in (batch, seq)
        # nn.CrossEntropyLoss expect to have (N, class) and (N,) as input
        # here we flatten logits and labels in order to apply nn.CrossEntropyLoss
        class_num = predicted.size(-1)
        #print('predicted[0]',predicted[0])
        loss, perm_idx, perm_list = self.objective(predicted, labels, lengths) #返回最小的permutation的loss 和 perm_idx
        # get the best label permutation
        label_perm = get_label_perm(labels, perm_idx, perm_list)   #确定计算loss时用到的label的permutation，同一个batch里可以用不同的permutation计算Loss，只取最小的，用来计算评分。其实也可以对predicted取对应的perm。
        # print(label_perm.shape)
        # exit()
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = calc_diarization_error(predicted, label_perm, lengths)

        if speech_scored > 0 and speaker_scored > 0 and num_frames > 0:
            SAD_MR, SAD_FR, MI, FA, CF, ACC, DER = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,    #这里DER指标不考虑重叠语音
                #DER就是三者百分比之和，
                #missed speech -- percentage of scored time for which a speech region is incorrectly marked as not containing speech
                #false alarm speech -- percentage of scored time for which a nonspeech region is incorrectly marked as containing speech
                #speaker error -- percentage of scored time for which the wrong speaker id is assigned within a speech region
            )
        else:
            SAD_MR, SAD_FR, MI, FA, CF, ACC, DER = 0, 0, 0, 0, 0, 0, 0
        # debug
        # print("SAD_MR {}, SAD_FR {}, MI {}, FA {}, CF {}, ACC {}, DER {}".format(SAD_MR, SAD_FR, MI, FA, CF, ACC, DER))
        records["loss"].append(loss.item())
        records["acc"] += [ACC]
        records["der"] += [DER]

        if mode == "test" and self.save_predictions:   #这段代码将处理后的模型预测结果以概率形式保存到hdf5文件中。 
            predict = predicted.data.cpu().numpy()
            predict = np.vstack(list(predict))
            predict = 1 / (1 + np.exp(-predict))
            outpath = os.path.join(self.score_dir, "predictions", rec_id + ".h5")
            with h5py.File(outpath, "w") as wf:
                wf.create_dataset("T_hat", data=predict)  #T_hat是名称
        return loss

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        """
        Args:
            mode: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml
                'dev' or 'test' :
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint

        """
        average_acc = torch.FloatTensor(records["acc"]).mean().item()
        average_der = torch.FloatTensor(records["der"]).mean().item()

        logger.add_scalar(
            f"diarization/{mode}-acc", average_acc, global_step=global_step
        )
        logger.add_scalar(
            f"diarization/{mode}-der", average_der, global_step=global_step
        )
        print("mode {} acc {} der {}".format(mode, average_acc, average_der))

        save_ckpt = []
        if mode == "dev" and average_acc > self.best_score:
            self.best_score = torch.ones(1) * average_acc
            save_ckpt.append(f"best-states-{mode}.ckpt")

        return save_ckpt
