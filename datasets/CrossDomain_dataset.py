from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from datasets.Kvasir_transform import *
import numpy as np
import torch
import torchvision

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_pth_txt, label_pth_txt, imgsize,data_split='train'):
        self.imgsize    = imgsize
        self.data_split = data_split
        self.images,self.gts = [],[]
        self.obtain_transform(data_split)
        with open(image_pth_txt, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')

        with open(label_pth_txt, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.gts.append(line)

        self.images = sorted(self.images)
        self.gts    = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

    def obtain_transform(self,data_split):
        if data_split == 'train':
            self.transform = transforms.Compose([
                Resize((self.imgsize, self.imgsize)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomRotation(90),
                RandomZoom((0.9, 1.1))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.imgsize, self.imgsize)),
                transforms.ToTensor()])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.imgsize, self.imgsize)),
                transforms.ToTensor()])



    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt    = self.binary_loader(self.gts[index])
        data  = {'image': image, 'segmentation': gt}
        if self.data_split == 'train':
            data = self.prepare_sample_train(data)
        else:
            data = self.prepare_sample_test(data)

        return data

    def __len__(self):
        return self.size



    def prepare_sample_train(self,data):

        data = self.transform(data)

        downresize4 = torchvision.transforms.Resize([56, 56])
        seg_downresize4 = np.array(downresize4(data['segmentation']))  # )Image.frombuffer

        data['seg_down4'] = transforms.ToTensor()(seg_downresize4)
        data['seg_down4'] = (data['seg_down4'] - data['seg_down4'].min()) / (
                    data['seg_down4'].max() - data['seg_down4'].min() + 1e-8)
        data['seg_down4'] = torch.where(data['seg_down4'] > 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

        data['image'] = transforms.ToTensor()(data['image'])
        data['segmentation'] = transforms.ToTensor()(data['segmentation'])
        data['segmentation'] = (data['segmentation'] - data['segmentation'].min()) / (
                    data['segmentation'].max() - data['segmentation'].min() + 1e-8)
        data['segmentation'] = torch.where(data['segmentation'] > 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        data['seg_down4'] = torch.squeeze(data['seg_down4'])
        data['segmentation'] = torch.squeeze(data['segmentation'])

        return data

    def prepare_sample_test(self,ori_data):

        image = self.transform(ori_data['image']).unsqueeze(0)
        gt = self.gt_transform(ori_data['segmentation']).unsqueeze(0)
        gt = (gt - gt.min()) / (gt.max() - gt.min())
        gt = torch.where(gt > 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

        data = {'name': '', 'image': image, 'segmentation': gt}
        return data


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')




def get_crossdomain_dataset(image_root, gt_root, imgsize,data_split='train'):

    dataset = PolypDataset(image_root, gt_root, imgsize,data_split)
    return dataset