import numpy as np
import torch
import torch.nn.functional as F

import torch.optim as optim

from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings
from typing import List

from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch import nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import cv2
#################### boundary iou #########################

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


#################### boundary iou #########################


def obtain_optimizer(cfg,net,):
    segment_dict = []
    backbone_dict = []

    for i, para in enumerate(net.parameters()):
        if i < 38:
            segment_dict.append(para)
        else:
            backbone_dict.append(para)

    seg_optimizer = optim.SGD(
        params=[
            {'params': backbone_dict, 'lr': cfg.TRAIN_LR},
            {'params': segment_dict, 'lr': 10 * cfg.TRAIN_LR}
        ],
        momentum=cfg.TRAIN_MOMENTUM)

    return seg_optimizer

def load_pretrained_dict(cfg,net):
    pretrained_dict = torch.load(cfg.TRAIN_CKPT)
    net_dict = net.state_dict()
    if cfg.TRAIN_GPUS <= 1:
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            k = k.replace('module.', '')
            new_pretrained_dict[k] = v

        pretrained_dict = new_pretrained_dict

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    return net

class LinearWarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

def adjust_lr(cfg,optimizer, itr, max_itr):
    now_lr = cfg.TRAIN_LR * (1 - (itr / (max_itr + 1)) ** cfg.TRAIN_POWER)
    optimizer.param_groups[0]['lr'] = now_lr
    optimizer.param_groups[1]['lr'] = 10 * now_lr
    return now_lr

### training utils  ###


def test_one_epoch(cfg,DATAloader, net,output_logtxt_path):
    #### start testing now
    Acc_array   = 0.
    Prec_array  = 0.
    Spe_array   = 0.
    Rec_array   = 0.
    IoU_array   = 0.
    Dice_array  = 0.
    HD_array    = 0.
    sample_num  = 0.
    b_iou_array = 0.
    result_list = []

    net.eval()
    with torch.no_grad():

        for i_batch, sample_batched in enumerate(DATAloader):
            name_batched   = sample_batched['name']
            [batch, channel, height, width] = sample_batched['image'].size()
            labels_batched = sample_batched['segmentation'].cpu().numpy()
            inputs_batched = sample_batched['image']  # _%f' % rate]
            inputs_batched = inputs_batched.cuda()
            predicts       = net(inputs_batched)
            result         = torch.argmax(predicts, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(batch):

                p          = result[i, :, :]
                l          = labels_batched[i, :, :]
                b_iou_item = boundary_iou(p, l, dilation_ratio=0.02)
                predict    = np.int32(p)
                gt         = np.int32(l)

                P  = np.sum((predict == 1)).astype(np.float64)
                T  = np.sum((gt == 1)).astype(np.float64)
                TP = np.sum((gt == 1) * (predict == 1)).astype(np.float64)
                TN = np.sum((gt == 0) * (predict == 0)).astype(np.float64)

                Acc  = (TP + TN) / (T + P - TP + TN)
                Prec = TP / (P + 1e-10)
                Spe  = TN / (P - TP + TN)
                Rec  = TP / T
                DICE = 2 * TP / (T + P)
                IoU  = TP / (T + P - TP)
                beta          = 2
                HD            = Rec * Prec * (1 + beta ** 2) / (Rec + beta ** 2 * Prec + 1e-10)
                Acc_array    += Acc
                Prec_array   += Prec
                Spe_array    += Spe
                Rec_array    += Rec
                Dice_array   += DICE
                IoU_array    += IoU
                HD_array     += HD
                b_iou_array  += b_iou_item
                sample_num   += 1

                result_list.append(
                    {'predict': np.uint8(p * 255), 'label': np.uint8(l * 255), 'IoU': IoU, 'name': name_batched[i]})

        Acc_score   = Acc_array   * 100 / sample_num
        Prec_score  = Prec_array  * 100 / sample_num
        Spe_score   = Spe_array   * 100 / sample_num
        Rec_score   = Rec_array   * 100 / sample_num
        Dice_score  = Dice_array  * 100 / sample_num
        IoUP        = IoU_array   * 100 / sample_num
        HD_score    = HD_array    * 100 / sample_num
        b_iou_score = b_iou_array * 100 / sample_num
        print(
            '%10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%\n' % (
                'Acc', Acc_score, 'Sen', Rec_score, 'Spe', Spe_score, 'Prec', Prec_score, 'Dice', Dice_score, 'Jac',
                IoUP, 'F2', HD_score, 'b_iou', b_iou_score))
        with open(output_logtxt_path, "a") as f:
            f.write(
                '%10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%\n' % (
                'Acc', Acc_score, 'Sen', Rec_score, 'Spe', Spe_score, 'Prec', Prec_score, 'Dice', Dice_score, 'Jac',
                IoUP, 'F2', HD_score))

        return Acc_score, Prec_score, Spe_score, Rec_score, Dice_score, IoUP, HD_score, b_iou_score