import torch
import numpy as np
import cv2

##################### Mix up #############################
def input_mixup(inputs1, inputs2, labels1, labels2, lambda1, i_batch):
    #####膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 3*3的正方形
    # kernel=np.ones((3,3),np.uint8)
    new_labels1 = torch.tensor(
        cv2.dilate(cv2.UMat(np.array(labels1.cpu().permute(1, 2, 0), dtype=np.uint8)), kernel).get()).permute(2, 0,
                                                                                                              1)  # [16,256,256]

    # new_labels1_in2 = torch.where(torch.sum(inputs2, 1).mul(new_labels1) == 0., torch.zeros_like(new_labels1), new_labels1)####避免旋转缩放等操作后前景不在图片中
    new_labels2 = torch.tensor(
        cv2.dilate(cv2.UMat(np.array(labels2.cpu().permute(1, 2, 0), dtype=np.uint8)), kernel).get()).permute(2, 0,
                                                                                                              1)
    # add_labels = new_labels1_in2 + new_labels2
    add_labels = new_labels1 + new_labels2

    #####inputs1中前景部分的颜色平均值
    x1 = torch.sum(inputs1.mul(new_labels1.unsqueeze(1).repeat(1, 3, 1, 1)), (2, 3))  # [16,3]
    y1 = torch.sum(new_labels1, (1, 2)).unsqueeze(1).repeat(1, 3).float()  # [16,3]
    mcolor_inputs1 = torch.where(y1 == 0, torch.zeros_like(y1), torch.div(x1, y1))
    mcolor_inputs1 = mcolor_inputs1.unsqueeze(2).unsqueeze(3).repeat(1, 1, inputs1.shape[2], inputs1.shape[3])

    #####inputs1中前景部分对应的inputs2里位置的颜色平均值
    pos_x1 = torch.sum(inputs2.mul(new_labels1.unsqueeze(1).repeat(1, 3, 1, 1)), (2, 3))
    pos_mcolor_inputs1 = torch.where(y1 == 0, torch.zeros_like(y1), torch.div(pos_x1, y1))
    pos_mcolor_inputs1 = pos_mcolor_inputs1.unsqueeze(2).unsqueeze(3).repeat(1, 1, inputs1.shape[2], inputs1.shape[3])

    # 对inputs1前景进行调整： 对应的inputs2部分的均值/inputs1前景部分均值*inputs1
    adjust_inputs1 = torch.where(mcolor_inputs1 == 0, torch.zeros_like(mcolor_inputs1),
                                 torch.div(pos_mcolor_inputs1, mcolor_inputs1)).mul(inputs1)
    only_inputs1 = torch.where(add_labels.mul(new_labels1).unsqueeze(1).repeat(1, 3, 1, 1) == 1, adjust_inputs1,
                               torch.zeros_like(inputs1))
    only_inputs2 = torch.where(add_labels.mul(new_labels2).unsqueeze(1).repeat(1, 3, 1, 1) == 1, inputs2,
                               torch.zeros_like(inputs2))
    overlap_inputs1and2 = torch.where(add_labels.unsqueeze(1).repeat(1, 3, 1, 1) == 2,
                                      adjust_inputs1.mul(0.5) + inputs2.mul(0.5), torch.zeros_like(inputs1))
    mixup_fg = only_inputs1 + only_inputs2 + overlap_inputs1and2


    mixup_inputs = torch.where(add_labels.unsqueeze(1).repeat(1, 3, 1, 1) == 0, inputs2, mixup_fg)

    return mixup_inputs


def label_mixup(labels1, labels2, label_down4, label_down4_2):
    mixup_labels = torch.where(labels1 + labels2 > 0, torch.ones_like(labels1), torch.zeros_like(labels1))
    mixup_labels_down4 = torch.where(label_down4 + label_down4_2 > 0, torch.ones_like(label_down4),
                                     torch.zeros_like(label_down4))
    return mixup_labels, mixup_labels_down4

##################### Mix up #############################