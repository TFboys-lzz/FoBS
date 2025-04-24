# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from model.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from model.backbone import build_backbone
from model.ASPP import ASPP



class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

        #self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # 偏置层  卷积核中所有元素要偏移的x、y坐标
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        # 权重学习层
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        #print(x.shape)
        offset = self.p_conv(x)  # [8,18,64,64]
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # 将offset放到网格上，也就是标定出每一个坐标位置
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # floor是向下取整
        q_lt = p.detach().floor()
        # 加一相当于向上取整，如果正好是整数的话，向上取整和向下取整就重合了
        q_rb = q_lt + 1

        # 将lt和rb限制在图像范围内，前N个代表x坐标， 后N个代表y坐标
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 获得lb和rt
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)


        # 插值的时候需要考虑一下padding对原始索引的影响
        # (b, h, w, N)
        # torch.lt() 逐元素比较input和other，即是否input < other
        # torch.rt() 逐元素比较input和other，即是否input > other
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding), p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        # 禁止反向传播
        mask = mask.detach()
        # p - (p - torch.floor(p))不就是torch.floor(p)呢。。。
        floor_p = p - (p - torch.floor(p))
        # 总的来说就是把超出图像的偏移量向下取整
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)  # 最终的插值操作
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        # 偏置点含有九个方向的偏置，_reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式，
        #  # 于是就可以用 3×3 stride=3 的卷积核进行 Deformable Convolution，
        #  # 它等价于使用 1×1 的正常卷积核（包含了这个点9个方向的 context）对原特征直接进行卷积。
        x_offset = self._reshape_x_offset(x_offset, ks)  # 8,304,192,192
        #print(x_offset.shape)
        #lap = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        lap = [[-1.0/8.0, -1.0/8.0, -1.0/8.0], [-1.0/8.0, 1, -1.0/8.0], [-1.0/8.0, -1.0/8.0, -1.0/8.0]]
        channels = x.shape[1]
        # 3*3 -> 1*3*3 -> 1*1*3*3 -> 1*channels*3*3
        kel = torch.tensor(lap, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3).to(device=torch.cuda.current_device())
        out = F.conv2d(x_offset, kel, stride=3, padding=1)
        #out = self.conv(x_offset)

        return out, offset

    # 求每个点的偏置方向
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    #求每个点的坐标
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 求最后的偏置后的点=每个点的坐标+偏置方向+偏置
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    # 求出p点周围四个点的像素
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w) 将图片压缩到1维，方便后面的按照index索引提取
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N) 这个目的就是将index索引均匀扩增到图片一样的h*w大小
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # 双线性插值法就是4个点再乘以对应与 p 点的距离。获得偏置点 p 的值，这个 p 点是 9 个方向的偏置所以最后的 x_offset 是 b×c×h×w×9
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    # _reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()  # [8,304,64,64,9]
        #print(x_offset.size())
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset







class ori_deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(ori_deeplabv3plus, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                dim_out=cfg.MODEL_ASPP_OUTDIM,
                rate=16//cfg.MODEL_OUTPUT_STRIDE,# 整除
                bn_mom = cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
                SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
                nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
                nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)


        self.para = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.para.data.fill_(1.0)



        self.offset_laplas_conv = DeformConv2d(inc=cfg.MODEL_ASPP_OUTDIM + cfg.MODEL_SHORTCUT_DIM, outc=cfg.MODEL_ASPP_OUTDIM + cfg.MODEL_SHORTCUT_DIM)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()





    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0, std=0.0001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, y_true=None, y=None, margin=None, feature_cat=None, phase='orig'):

        # testing
        if y_true == None:
            x_bottom = self.backbone(x)
            layers = self.backbone.get_layers()
            feature_aspp = self.aspp(layers[-1])
            feature_aspp = self.dropout1(feature_aspp)
            feature_aspp = self.upsample_sub(feature_aspp)
            feature_shallow = self.shortcut_conv(layers[0])
            feature_cat = torch.cat([feature_aspp,feature_shallow],1)

            result = self.cat_conv(feature_cat)
            temp_result = self.cls_conv(result)
            result = self.upsample4(temp_result)

            offset_1, offset = self.offset_laplas_conv(feature_cat)


            return feature_cat, result, offset, self.para
        # training
        else:
            x_bottom = self.backbone(x)
            layers = self.backbone.get_layers()
            ############ features ###############
            feature_aspp = self.aspp(layers[-1])
            feature_aspp = self.dropout1(feature_aspp)
            feature_aspp = self.upsample_sub(feature_aspp)
            feature_shallow = self.shortcut_conv(layers[0])
            feature_cat_1 = torch.cat([feature_aspp, feature_shallow], 1)  # print(feature_cat.shape)  # [16, 304, 64, 64]

            ############ path one ori result ###############
            result_1 = self.cat_conv(feature_cat_1)
            result_temp_1 = self.cls_conv(result_1)
            result_1 = self.upsample4(result_temp_1)


            ############ att of path1 ###############
            # print(y.shape)  # [16, 64, 64]
            # print(result_path_f1_temp.shape)  # [16, 2, 64, 64]
            result_temp_1_soft_channel1 = nn.Softmax(dim=1)(result_temp_1)[:, 1, :, :]
            zeros_mat = torch.zeros_like(y).float()
            temp_guide = torch.abs(y - result_temp_1_soft_channel1)
            att_guide_1 = torch.where(temp_guide>margin, temp_guide, zeros_mat)  # print(att_guide.shape)  # [16, 64, 64]


            ones_mat = torch.ones_like(y).float()
            num = torch.sum(torch.where(temp_guide>margin, ones_mat, zeros_mat))
            avg_att = torch.sum(temp_guide)/num

            ############ refine backbone features ###############
            offset_1, _ = self.offset_laplas_conv(feature_cat_1)
            coe = torch.exp(self.para * torch.unsqueeze(att_guide_1, 1))-1
            refine_feature_cat_2 = feature_cat_1 - offset_1 * coe
            ############ refine path2 result ###############

            refine_result_2 = self.cat_conv(refine_feature_cat_2)
            refine_result_2 = self.cls_conv(refine_result_2)
            refine_result_2 = self.upsample4(refine_result_2)



            return feature_cat_1, result_1, refine_feature_cat_2, refine_result_2, self.para, avg_att, coe
