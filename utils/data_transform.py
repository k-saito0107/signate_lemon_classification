import numpy as np
import torch
from torchvision import transforms
from utils.data_augumentation import Compose, Enhance, Resize, Scale, Normalize_Tensor


class DataTransform():
    def __init__(self, width, height, color_mean, color_std):
        self.data_transform = {
            'train':Compose([
                Resize(width, height),
                Enhance(factor=[0.7, 1.3]),
                Scale(scale=[1.0, 1.3]),
                transforms.RandomAffine(degrees=(-10, 10)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                Normalize_Tensor(color_mean, color_std)
            ]),
            'val':Compose([
                Resize(width, height),
                Normalize_Tensor(color_mean, color_std)
            ])
        }
    
    def __call__(self, phase, img):
        return self.data_transform[phase](img)