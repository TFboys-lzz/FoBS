import torch.nn as nn
import torch
import torchvision


###################### Loss Function  ##########################

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        #    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        #    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        #    return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        return self.TVLoss_weight * (h_tv[:, :, :, 1:] + w_tv[:, :, 1:, :]).sum(dim=1)

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def Overlap_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        #true [16,256,256]
        #logits [16,2,256,256]
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """

    num_classes = logits.shape[1]
    # print(num_classes)  #2
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1).to(true.device)
        true_1_hot = true_1_hot[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes + 1).to(true.device)
        true_1_hot = true_1_hot[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = logits[:, 1, :, :].cuda()  # [b,h,w]
    true_1_hot = true_1_hot.type(logits.type())[:, 1, :, :].cuda()  # [b,h,w]
    dims = (0,) + tuple(range(2, true.ndimension()))
    #    intersection = torch.sum(probas * true_1_hot, dims)
    #    cardinality = torch.sum(probas + true_1_hot, dims)
    # 计算交集
    intersection = torch.sum(probas * true_1_hot, dim=(1, 2))
    # compute |X|+|Y|
    cardinality = torch.sum(probas + true_1_hot, dim=(1, 2))
    # compute union set
    union = cardinality - intersection
    # fg loss
    jacc_loss = (intersection / (union + eps)).mean()
    dice_loss = (2 * intersection / (cardinality)).mean()
    return (1 - jacc_loss), (1 - dice_loss)


def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -torch.log(inputs)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.mean(torch.mul(log_likelihood, target))
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


def MSE(inputs, target, reduction='average'):
    MSE = torch.nn.MSELoss(reduce=False, size_average=False)
    loss = MSE(inputs, target).sum(dim=1)
    if reduction == 'average':
        loss = torch.mean(loss)
    else:
        loss = torch.sum(loss)
    return loss


def diffScore(labels_batched, predicts_batched, refine_predicts_batched, mode='weight'):

    '''

    :param labels_batched:    ground truth labels [b,h,w]
    :param predicts_batched:  basic branch predictions [b,2,h,w]
    :param refine_predicts_batched: refined branch predictions [b,2,h,w]
    :param mode:
    :return:
    '''



    # obtain the foreground prediction of two branches
    predicts_batched_soft_channel1 = nn.Softmax(dim=1)(predicts_batched)[:, 1, :, :].cuda()
    refine_predicts_batched_soft_channel1 = nn.Softmax(dim=1)(refine_predicts_batched)[:, 1, :, :].cuda()

    # compute the positive space and negative space
    pos_space = torch.abs(labels_batched - predicts_batched_soft_channel1).cuda()
    neg_space = 1 - pos_space


    pos_ones_mat = torch.ones_like(labels_batched).cuda()
    neg_ones_mat = pos_ones_mat * (-1)
    state = torch.where(labels_batched > 0, pos_ones_mat, neg_ones_mat).cuda()  # i.e., T in paper fg: 1, bg: -1

    # refine_p2减p1求得初始的score, 并用state校正状    # 校正取值范围为[-1,1]  负的代表预测能力退化了，正的代表预测能力提升了
    # compute the transformed bias \hat{S} in the paper,
    # the value > 0, the prediction ability is improved
    # the value < 0, the prediction ability is decreased
    ori_score = ((refine_predicts_batched_soft_channel1 - predicts_batched_soft_channel1) * state).cuda()

    if mode == 'loss':

        zeros           = torch.zeros_like(ori_score).cuda()
        score_to_loss   = torch.where(ori_score >= 0, zeros, ori_score / (neg_space + 1e-7))
        diff_score_loss = torch.mean(torch.abs(score_to_loss)) * 10
        return diff_score_loss

    elif mode == 'weight':

        # as the classification weight

        ori_score_to_weight = torch.where(ori_score >= 0, ori_score / (pos_space + 1e-7),
                                          ori_score / (neg_space + 1e-7))   #  normalization
        score_to_weight     = 1 - (ori_score_to_weight + 1.) / 2.

        return score_to_weight, ori_score_to_weight
