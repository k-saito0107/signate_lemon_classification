import numpy as np
from PIL import Image
import cv2
import random
import os
import os.path as osp
import csv

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torchvision.datasets import ImageFolder
import pandas as pd
import torch.optim as optim
from glob import glob
#from efficientnet_pytorch import EfficientNet

from utils.data_transform import DataTransform
from utils.make_dataset import Make_Dataset
from utils.train import train




def main():
    root_path = '/kw_resources/signate/signate_lemon_classification/data'

    img_path = osp.abspath(root_path + '/train_images/')
    img_list = sorted(glob(osp.join(img_path, '*.jpg')))
    #val_img_path = osp.abspath(root_path + '/test_images/')
    #val_img_list = sorted(glob(osp.join(val_img_path, '*.jpg')))
    num = len(img_list)
    theta = int(0.8 * num)
    random.shuffle(img_list)
    train_img_list = img_list[:theta]
    val_img_list = img_list[theta:]
    print(len(train_img_list))
    print(len(val_img_list))

    csv_path = '/kw_resources/signate/signate_lemon_classification/data/train_images.csv'
    data = []
    label_dic = {}
    with open(csv_path, newline='', encoding='utf_8_sig') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    for i in range(len(data)):
        label_dic[data[i]['id']] = data[i]['class_num']


    color_mean = (0.5, 0.5, 0.5)
    color_std = (0.5, 0.5, 0.5)
    width = 448
    height = 448

    train_dataset = Make_Dataset(train_img_list, label_dic, DataTransform(width=width, height=height, color_mean=color_mean,color_std=color_std), phase='train')
    val_dataset = Make_Dataset(val_img_list, label_dic, DataTransform(width=width, height=height, color_mean=color_mean,color_std=color_std), phase='val')
    batchsize = 16
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batchsize,shuffle=True)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)
    '''
    model = EfficientNet.from_pretrained('efficientnet-b6')
    num_ftrs = model._fc.in_features　　#全結合層の名前は"_fc"となっています
    model._fc = nn.Linear(num_ftrs, 4)
    '''
    #model
    use_pretrained = True
    model = models.resnet50(pretrained=use_pretrained)
    model.fc = nn.Linear(in_features=512, out_features=4)

    num_epoch = 300
    up_model = train(model, num_epoch, train_loader, val_loader)


if __name__ == '__main__':
    main()
    print('finish')