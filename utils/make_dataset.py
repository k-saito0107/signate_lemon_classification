import torch
import torch.utils.data as data
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

class Make_Dataset(data.Dataset):
    def __init__(self, img_path,label_dict, img_transform, phase):
        self.img_path = img_path
        self.label_dict = label_dict
        self.img_transform = img_transform
        self.phase = phase

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        img_file_path = self.img_path[index]
        img = Image.open(img_file_path)
        img = self.img_transform(self.phase, img)
        #print(img_file_path)

        img_name = img_file_path.split('/')[-1]
        label = self.label_dict[img_name]

        return img, label
    

