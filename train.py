import torch
import random
import yaml
import os
import torch.nn as nn
from datasets.generate_dataset import generate_dataset
from model.generateNet import generate_net
from torch.utils.data import DataLoader
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.SSIM import *

from utils.train_utils import *
from utils.mixup_utils import *
from utils.loss import *
from utils.SSIM import SSIM
import argparse

import matplotlib
matplotlib.use('agg')



def get_params(model, key):
    for m in model.named_modules():
        if key == '1x':
            if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file)
    torch.save(model, new_file)
    print('%s has been saved'%new_file)


def train_net(cfg):



    cfg.MODEL_SAVE_DIR = os.path.join(cfg.ROOT_DIR,'out', cfg.EXP_NAME,'model')
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    with open(os.path.join(cfg.MODEL_SAVE_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    output_logtxt_path = cfg.MODEL_SAVE_DIR + '/output_log.txt'



    dataset    = generate_dataset(cfg.DATA_NAME, cfg, 'train')
    dataloader = DataLoader(dataset,
                            batch_size=cfg.TRAIN_BATCHES,
                            shuffle=cfg.TRAIN_SHUFFLE,
                            num_workers=cfg.DATA_WORKERS,
                            drop_last=True)

    dataset_mixup    = generate_dataset(cfg.DATA_NAME, cfg, 'train')
    dataloader_mixup = DataLoader(dataset_mixup,
                                  batch_size=cfg.TRAIN_BATCHES,
                                  shuffle=cfg.TRAIN_SHUFFLE,
                                  num_workers=cfg.DATA_WORKERS,
                                  drop_last=True)

    test_dataset    = generate_dataset(cfg.DATA_NAME, cfg, 'test')
    test_dataloader = DataLoader(test_dataset,
                                      batch_size=cfg.TEST_BATCHES,
                                      shuffle=False,
                                      num_workers=cfg.DATA_WORKERS)


    net = generate_net(cfg)

    print('Use %d GPU' % cfg.TRAIN_GPUS)
    device = torch.device(0)
    if cfg.TRAIN_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.to(device)

    if cfg.TRAIN_CKPT:

        net = load_pretrained_dict(cfg,net)
        print('load pretrained model from {}'.format(cfg.TRAIN_CKPT))

    seg_optimizer = obtain_optimizer(cfg,net)
    scheduler = LinearWarmupCosineAnnealingLR(seg_optimizer, warmup_epochs=cfg.WARMUP_EPOCHS, max_epochs=cfg.TRAIN_EPOCHS)

    for i in range(0, cfg.TRAIN_MINEPOCH):
        scheduler.step()

    itr               = cfg.TRAIN_MINEPOCH  * len(dataloader)
    max_itr           = cfg.TRAIN_EPOCHS_lr * len(dataloader)

    best_jacc         = 0.
    best_epoch        = 0
    best_biou         = 0

    ########################### define loss function ######################

    cr_criterion      = nn.CrossEntropyLoss(ignore_index=255)
    none_cr_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    init_ssim_loss = SSIM()


    for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):

        running_loss                    = 0.0
        seg_overlap_running_loss        = 0.0
        refine_running_loss             = 0.0
        refine_seg_overlap_running_loss = 0.0
        ssim_running_loss               = 0.0


        mixup_running_loss                    = 0.0
        mixup_seg_overlap_running_loss        = 0.0
        mixup_refine_running_loss             = 0.0
        mixup_refine_seg_overlap_running_loss = 0.0
        mixup_ssim_running_loss               = 0.0


        net.train()


        for i_batch, (sample_batched, mixup_batched) in enumerate(zip(dataloader, dataloader_mixup)):

            now_lr = adjust_lr(cfg,seg_optimizer, itr, max_itr)

            #################################### read data ###########################################
            inputs_batched_cpu, labels_batched_cpu, label_down4_cpu = sample_batched['image'], sample_batched['segmentation'].long(), \
                                                              sample_batched['seg_down4'].long()

            inputs_batched_gpu  = inputs_batched_cpu.cuda()
            labels_batched_gpu  = labels_batched_cpu.cuda()  # label  [b,h,w]
            label_down4_gpu     = label_down4_cpu.cuda()

            ###################################################################### train orign samples ######################################################################
            feature_batched_1, predicts_batched_1, refine_feature_batched_2, refine_predicts_batched_2, avg_att, beta2, coe, offcoe = net(
                inputs_batched_gpu, labels_batched_gpu, label_down4_gpu, cfg.MARGIN)

            ############################   diff score loss ###########################################
            # obtain the weight in eq (11) of the paper
            score_to_weight_2, ori_score_to_weight = diffScore(labels_batched_gpu, predicts_batched_1,
                                                               refine_predicts_batched_2, mode='weight')

            ############################### segmentation loss of basic branch ########################################
            loss_1                          = cr_criterion(predicts_batched_1, labels_batched_gpu)
            soft_predicts_batched_1         = nn.Softmax(dim=1)(predicts_batched_1)
            seg_jac_loss_1, seg_dice_loss_1 = Overlap_loss(labels_batched_gpu, soft_predicts_batched_1, eps=1e-7)

            ############################### refine segmentation loss ########################################
            ori_ce_loss_2  = none_cr_criterion(refine_predicts_batched_2, labels_batched_gpu)
            refine_loss_2                                 = torch.mean(torch.mul(score_to_weight_2, ori_ce_loss_2)) #  lscr loss
            refine_soft_predicts_batched_2                = nn.Softmax(dim=1)(refine_predicts_batched_2)
            refine_seg_jac_loss_2, refine_seg_dice_loss_2 = Overlap_loss(labels_batched_gpu, refine_soft_predicts_batched_2,
                                                                         eps=1e-7)

            ############################ feature ssim loss #################################
            ssim_loss     = (1 - init_ssim_loss(feature_batched_1, refine_feature_batched_2))

            ############################ update network #################################
            total_loss = loss_1 + seg_dice_loss_1 + cfg.ssim_weight * ssim_loss + cfg.refine_weight * (refine_loss_2 + refine_seg_dice_loss_2)
            seg_optimizer.zero_grad()
            total_loss.backward()
            seg_optimizer.step()



            running_loss                    += loss_1
            seg_overlap_running_loss        += seg_jac_loss_1
            refine_running_loss             += refine_loss_2
            refine_seg_overlap_running_loss += refine_seg_dice_loss_2
            ssim_running_loss               += ssim_loss


            ###################################################################### train mixup samples ######################################################################
            if epoch >= cfg.Mixup_start_epoch:

                ################### prepare data ##########################
                inputs_batched2_cpu, labels_batched2_cpu, label_down4_2_cpu = mixup_batched['image'], mixup_batched[
                    'segmentation'], mixup_batched['seg_down4']

                #########################################   mixup data   #################################
                alpha = cfg.Alpha
                random_lambda = np.random.beta(alpha, alpha)

                ##### obtain input cpu tensor ##################


                mixup_inputs = input_mixup(inputs_batched_cpu, inputs_batched2_cpu, labels_batched_cpu, labels_batched2_cpu,random_lambda,i_batch)
                mixup_labels, mixup_labels_down4 = label_mixup(labels_batched_cpu, labels_batched2_cpu, label_down4_cpu,
                                                               label_down4_2_cpu)


                mixup_inputs       = mixup_inputs.cuda()
                mixup_labels       = mixup_labels.cuda()
                mixup_labels_down4 = mixup_labels_down4.cuda()


                #########################################  train mixup #################################
                mixup_feature_batched_1, mixup_predicts_batched_1, mixup_refine_feature_batched_2, mixup_refine_predicts_batched_2, mixup_avg_att, mixup_beta2, mixup_coe, mixup_offcoe = net(
                    mixup_inputs, mixup_labels, mixup_labels_down4, cfg.MARGIN)


                ############################diff score loss ###########################################
                mixup_score_to_weight_2, mixup_ori_score_to_weight = diffScore(mixup_labels, mixup_predicts_batched_1,
                                                                               mixup_refine_predicts_batched_2,
                                                                               mode='weight')
                ############################### segmentation loss ########################################
                mixup_loss_1                  = cr_criterion(mixup_predicts_batched_1, mixup_labels)
                mixup_soft_predicts_batched_1 = nn.Softmax(dim=1)(mixup_predicts_batched_1)
                mixup_seg_jac_loss_1, mixup_seg_dice_loss_1 = Overlap_loss(mixup_labels, mixup_soft_predicts_batched_1,
                                                                           eps=1e-7)

                ############################## refine segmentation loss ##################################
                mixup_ori_ce_loss_2 = none_cr_criterion(mixup_refine_predicts_batched_2, mixup_labels)
                mixup_refine_loss_2 = torch.mean(torch.mul(mixup_score_to_weight_2, mixup_ori_ce_loss_2))
                mixup_refine_soft_predicts_batched_2 = nn.Softmax(dim=1)(mixup_refine_predicts_batched_2)
                mixup_refine_seg_jac_loss_2, mixup_refine_seg_dice_loss_2 = Overlap_loss(mixup_labels,
                                                                                         mixup_refine_soft_predicts_batched_2,
                                                                                         eps=1e-7)
                ############################## refine segmentation loss ##################################


                ############################ feature sim or l2(MSE) loss #################################
                mixup_ssim_loss = (1 - init_ssim_loss(mixup_feature_batched_1, mixup_refine_feature_batched_2))
                mixup_seg_loss  = mixup_loss_1 + mixup_seg_dice_loss_1 + cfg.ssim_weight * mixup_ssim_loss + cfg.refine_weight * (mixup_refine_loss_2 + mixup_refine_seg_dice_loss_2)

                seg_optimizer.zero_grad()
                mixup_seg_loss.backward()
                seg_optimizer.step()

                mixup_running_loss += mixup_loss_1
                mixup_seg_overlap_running_loss += mixup_seg_jac_loss_1
                mixup_refine_running_loss += mixup_refine_loss_2
                mixup_refine_seg_overlap_running_loss += mixup_refine_seg_jac_loss_2
                mixup_ssim_running_loss += mixup_ssim_loss


            itr += 1
            # scheduler.step()

        # print('##### learning rate {} #####'.format(now_lr))
        print('Ori  :epoch:%d/%d\tCE:%g \tdice:%g \tre_CE:%g \tre_jac:%g \tssim:%g' % (
                epoch, cfg.TRAIN_EPOCHS, running_loss / i_batch, seg_overlap_running_loss / i_batch,
                refine_running_loss / i_batch, refine_seg_overlap_running_loss / i_batch, ssim_running_loss / i_batch))


        if epoch >= cfg.Mixup_start_epoch:
            print(
                'Mixup:epoch:%d/%d\tCE:%g \tdice:%g \tre_CE:%g \tre_jac:%g \tssim:%g' % (
                    epoch, cfg.TRAIN_EPOCHS,mixup_running_loss / i_batch, mixup_seg_overlap_running_loss / i_batch,
                    mixup_refine_running_loss / i_batch,
                    mixup_refine_seg_overlap_running_loss / i_batch, mixup_ssim_running_loss / i_batch))



        #### start testing now
        if  epoch % 10 == 0:
            torch.cuda.empty_cache()

            print('######  test  ##########')
            Acc_score, Rec_score, Spe_score, Prec_score, Dice_score, IoUP, HD_score, b_iou_score = test_one_epoch(cfg,test_dataloader, net, output_logtxt_path)
            print('######  test  ##########')

            if Dice_score > best_jacc:
                model_snapshot(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'epoch%d_dice%.3f_biou%.3f.pth'%(epoch,Dice_score,b_iou_score)),
                               old_file=os.path.join(cfg.MODEL_SAVE_DIR,'epoch%d_dice%.3f_biou%.3f.pth'%(epoch,Dice_score,b_iou_score)))
                best_jacc = Dice_score
                best_epoch = epoch
                best_biou = b_iou_score


    torch.cuda.empty_cache()

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    torch.cuda.manual_seed_all(seed) #all gpus
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='./config/config_kvasir.yaml', help='The path resume from checkpoint')

    args = parser.parse_args()

    with open(args.cfg_path, 'r') as f:

        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)
    print(args)


    # setup_seed(3407)
    train_net(args)
