# ----------------------------------------
# Written by Xiaoqing GUO
# ----------------------------------------

from __future__ import print_function, division
import os
import pandas as pd
import cv2
import multiprocessing
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *

from PIL import Image
import torchvision

class CVCDataset(Dataset):

    def __init__(self, dataset_name, cfg, data_split):

        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.DATA_ROOT_DIR,'data')
        #print(self.root_dir)
        self.dataset_dir = os.path.join(self.root_dir,dataset_name)
        self.rst_dir = os.path.join(self.root_dir,'results',dataset_name,'Segmentation')
        self.eval_dir = os.path.join(self.root_dir,'eval_result',dataset_name,'Segmentation')
        self.data_split = data_split  #train/test
        self.img_dir = os.path.join(self.dataset_dir, 'images')
        self.ann_dir = os.path.join(self.dataset_dir, 'labels')
        self.seg_dir = os.path.join(self.dataset_dir, 'labels')
        self.set_dir = os.path.join(self.root_dir, dataset_name)
        file_name = None


        file_name = self.set_dir+'/'+self.data_split+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list      = df['filename'].values
        self.rescale        = None
        self.centerlize     = None
        self.randomcrop     = None
        self.randomflip     = None
        self.randomrotation = None
        self.randomscale    = None
        self.randomhsv      = None
        self.multiscale     = None
        self.totensor       = ToTensor()

        self.cfg = cfg



        if cfg.DATA_RESCALE > 0:  # 256
            self.rescale = Rescale(cfg.DATA_RESCALE,fix=False)
        if 'train' in self.data_split:
            if cfg.DATA_RANDOMCROP > 0:  # 384
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:  # 180
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:  # 1.25
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:  # 0.5
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:  # 10
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        else:
            self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)  # [1.0]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        name = self.name_list[idx].split()[0]
        img_file = self.img_dir + '/' + name 
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        if 'train' in self.data_split:

            sample = self.preprocess_train(idx, sample)

        else:
            sample = self.preprocess_test(idx, sample)

        if 'segmentation' in sample.keys():
            sample['mask'] = sample['segmentation'] < self.cfg.MODEL_NUM_CLASSES
            t = sample['segmentation']
            t[t >= self.cfg.MODEL_NUM_CLASSES] = 0
            sample['segmentation_onehot']=onehot(np.int32(t),self.cfg.MODEL_NUM_CLASSES)

        downresize4           = torchvision.transforms.Resize([64, 64])
        seg_downresize4       = np.array(downresize4(Image.fromarray(sample['segmentation'])))
        (T4, seg_downresize4) = cv2.threshold(seg_downresize4, 0, 255, cv2.THRESH_BINARY)
        sample['seg_down4']   = seg_downresize4 / 255.

        sample                = self.totensor(sample)



        return sample


    def preprocess_train(self, idx,sample):

        seg_file = self.seg_dir + '/' + self.name_list[idx].split()[0]  # label file

        segmentation = np.array(Image.open(seg_file))

        #    print(segmentation.shape,np.unique(segmentation))
        (T, segmentation) = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY)
        sample['segmentation'] = segmentation / 255.

        if self.cfg.DATA_RANDOM_H > 0 or self.cfg.DATA_RANDOM_S > 0 or self.cfg.DATA_RANDOM_V > 0:  # 10
            sample = self.randomhsv(sample)
        if self.cfg.DATA_RANDOMFLIP > 0:  #
            sample = self.randomflip(sample)
        if self.cfg.DATA_RANDOMROTATION > 0:
            sample = self.randomrotation(sample)
        if self.cfg.DATA_RANDOMSCALE != 1:
            sample = self.randomscale(sample)
        if self.cfg.DATA_RANDOMCROP > 0:
            sample = self.randomcrop(sample)
        if self.cfg.DATA_RESCALE > 0:
            # sample = self.centerlize(sample)
            sample = self.rescale(sample)

        return sample

    def preprocess_test(self, idx,sample):

        seg_file = self.seg_dir + '/' + self.name_list[idx].split()[0]
        segmentation = np.array(Image.open(seg_file))
        (T, segmentation) = cv2.threshold(segmentation, 10, 255, cv2.THRESH_BINARY)
        sample['segmentation'] = segmentation / 255.
        if self.cfg.DATA_RESCALE > 0:
            sample = self.rescale(sample)
        sample = self.multiscale(sample)

        return sample





    def do_matlab_eval(self, model_id):
        import subprocess
        path = os.path.join(self.root_dir, 'VOCcode')
        eval_filename = os.path.join(self.eval_dir,'%s_result.mat'%model_id)
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; VOCinit; '
        cmd += 'VOCevalseg(VOCopts,\'{:s}\');'.format(model_id)
        cmd += 'accuracies,avacc,conf,rawcounts = VOCevalseg(VOCopts,\'{:s}\'); '.format(model_id)
        cmd += 'save(\'{:s}\',\'accuracies\',\'avacc\',\'conf\',\'rawcounts\'); '.format(eval_filename)
        cmd += 'quit;"'

        print('start subprocess for matlab evaluation...')
        print(cmd)
        subprocess.call(cmd, shell=True)
    


    def do_python_eval(self, model_id):
        IoU_array = 0.
        sample_num = 0.
        predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.data_split))
        gt_folder = self.seg_dir
        TP = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        P = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        T = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        for idx in range(len(self.name_list)):
         #   print('%d/%d'%(idx,len(self.name_list)))
            name_image = self.name_list[idx].split()[0]
            name_seg = self.name_list[idx].split()[0]
            predict_file = os.path.join(predict_folder,'%s'%name_image)
            gt_file = os.path.join(gt_folder,'%s'%name_seg)
            predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
            gt = np.array(Image.open(gt_file))
            (_, predict) = cv2.threshold(predict, 0, 255, cv2.THRESH_BINARY)
            predict = np.int32(predict/255.)
            gt = np.int32(gt/255.)
            cal = gt<255
            mask = (predict==gt) * cal
            #print(np.unique(predict), np.unique(gt), np.unique(cal), np.unique(mask))
                
            for i in range(self.cfg.MODEL_NUM_CLASSES):
                P[i] = np.sum((predict==i))
                T[i] = np.sum((gt==i))
                TP[i] = np.sum((gt==i)*mask)
            TP = TP.astype(np.float64)
            T = T.astype(np.float64)
            P = P.astype(np.float64)
            IoU = TP/(T+P-TP)
            IoU_array += IoU
            sample_num += 1
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                print('%15s:%7.3f%%'%('background',IoU_array[i]*100/sample_num))
            else:
                print('%15s:%7.3f%%'%(self.categories[i-1],IoU_array[i]*100/sample_num))
        miou = np.mean(IoU_array/sample_num)
        print('==================================')
        print('%15s:%7.3f%%'%('mIoU',miou*100))
