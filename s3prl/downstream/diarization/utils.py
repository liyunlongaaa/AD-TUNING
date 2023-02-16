# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the speaker diarization dataset ]
#   Source       [ Refactored from https://github.com/hitachi-speech/EEND ]
#   Author       [ Jiatong Shi ]
#   Copyright    [ Copyleft(c), Johns Hopkins University ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import torch
import numpy as np
from itertools import permutations


# compute mask to remove the padding positions
def create_length_mask(length, max_len, num_output, device):
    batch_size = len(length)
    mask = torch.zeros(batch_size, max_len, num_output)
    for i in range(batch_size):
        mask[i, : length[i], :] = 1
    mask = mask.to(device)
    return mask


# compute loss for a single permutation
def pit_loss_single_permute(output, label, length):
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    mask = create_length_mask(length, label.size(1), label.size(2), label.device) #（b,t,2）因为可能同时说话，所以是一个多标签问题【1，1】表示同时
    #print(mask.shape)
    loss = bce_loss(output, label)   #（b,t,2）
    #print(loss.shape)
    loss = loss * mask
    loss = torch.sum(torch.mean(loss, dim=2), dim=1)
    loss = torch.unsqueeze(loss, dim=1)
    return loss


def pit_loss(output, label, length):
    num_output = label.size(2)
    device = label.device
    permute_list = [np.array(p) for p in permutations(range(num_output))]
    #print("permute_list",permute_list)
    loss_list = []
    for p in permute_list:
        label_perm = label[:, :, p]
        #print("lp",label_perm.shape, label_perm)
        loss_perm = pit_loss_single_permute(output, label_perm, length) #(b,1)
        loss_list.append(loss_perm)
    loss = torch.cat(loss_list, dim=1) #(b,2) 2是permute的个数 n！
    #print("loss",loss.shape, loss)
    min_loss, min_idx = torch.min(loss, dim=1)
    #print(min_loss, min_idx)   #计算loss可以取不同的permute?
    loss = torch.sum(min_loss) / torch.sum(length.float().to(device))
    return loss, min_idx, permute_list


def get_label_perm(label, perm_idx, perm_list):
    batch_size = len(perm_idx)
    label_list = []
    for i in range(batch_size):
        label_list.append(label[i, :, perm_list[perm_idx[i]]].data.cpu().numpy())
    return torch.from_numpy(np.array(label_list)).float()


def calc_diarization_error(pred, label, length):
    (batch_size, max_len, num_output) = label.size()
    # mask the padding part
    mask = np.zeros((batch_size, max_len, num_output))
    for i in range(batch_size):
        mask[i, : length[i], :] = 1

    # pred and label have the shape (batch_size, max_len, num_output)
    label_np = label.data.cpu().numpy().astype(int)
    pred_np = (pred.data.cpu().numpy() > 0).astype(int)  #0，1标签化， >0表示有声音？ 感觉怪怪的，downExpert激活函数也不是tanh
    label_np = label_np * mask
    pred_np = pred_np * mask
    length = length.data.cpu().numpy()

    # compute speech activity detection error
    n_ref = np.sum(label_np, axis=2)         #一段话是否有声音(>0)则有 
    n_sys = np.sum(pred_np, axis=2)
    speech_scored = float(np.sum(n_ref > 0))     #有人说话的总长度
    speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))  #有声音但没预测出来
    speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

    # compute speaker diarization error
    speaker_scored = float(np.sum(n_ref))       #说话总段数
    speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0))) #有人声音但没预测出来
    speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))

    n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)  #这两个有点牛逼 [0, 1] [1, 0] (或反过来才)算speaker_error
    speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
    # 上面这两段代码是在计算模型的预测结果与标签之间的差异（即错误）。

    # 第一行代码使用numpy的逻辑运算函数和sum函数计算两个布尔数组（label_np和pred_np）的逻辑与（AND）结果，然后沿着第三个维度（axis=2）对结果求和，最终得到一个表示真实标签和模型预测结果中同时为1的元素个数的矩阵。这个矩阵称为交集计数矩阵，用于衡量模型预测的说话人分割中正确的部分。具体来说，逻辑与操作将两个数组的值同时为1的位置设置为True，其余位置设置为False，求和后得到的结果就是两个数组同时为1的元素个数。

    # 第二行代码首先使用numpy的minimum函数计算两个标量的最小值，然后使用numpy的sum函数计算这些最小值的和，最终得到一个标量表示标签中的说话人数量与模型预测结果中的说话人数量之间的最小值的总和。这个值用于衡量模型预测的说话人分割中的错误部分。

    # 综上，这两行代码计算了模型预测结果中正确的说话人分割数量以及错误的说话人分割数量，用于评估模型在语音分割任务上的性能。
    correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
    num_frames = np.sum(length)
    return (
        correct,
        num_frames,
        speech_scored,
        speech_miss,
        speech_falarm,
        speaker_scored,
        speaker_miss,
        speaker_falarm,
        speaker_error,
    )
