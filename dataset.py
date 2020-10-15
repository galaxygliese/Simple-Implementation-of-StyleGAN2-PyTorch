#-*- coding:utf-8 -*-

from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


class Real_Data_Generator(Dataset):
    def __init__(self, datafolder, transforms=None, limit=10000):
        self.limit = limit
        self.datafolder = datafolder
        self.imgfiles = glob(os.path.join(datafolder, '*'))[:limit]
        self.transforms = transforms

    def crop_square(self, img):
        if len(img.shape) > 2:
            H, W, _ = img.shape
        else:
            H, W = img.shape
        S = W if W < H else H
        cropped = img[H//2-S//2:H//2+S//2, W//2-S//2:W//2+S//2, :] if len(img.shape) > 2 else img[H//2-S//2:H//2+S//2, W//2-S//2:W//2+S//2]
        return cropped

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, index):
        imgfile = self.imgfiles[index]
        img = np.array(Image.open(imgfile))
        img = self.crop_square(img)
        if self.transforms is not None:
            img = Image.fromarray(img)
            img = self.transforms(img)
        return img

if __name__ == '__main__':
    folder = '/home/galaxygliese/Desktop/Datasets/CelebA/img_align_celeba'
    composed = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    dataset = Real_Data_Generator(folder, transforms=composed)
    print(len(dataset))
    img = dataset[2]
    print(img.shape, torch.max(img), torch.min(img))
    img = img.data.permute(1,2,0).numpy()
    plt.imshow(img)
    plt.show()
