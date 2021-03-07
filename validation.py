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
import glob
#from efficientnet_pytorch import EfficientNet

from utils.data_transform import DataTransform
from utils.make_dataset import Make_Dataset

def main():
    img_path = glob.glob('./data/test_images/*.jpg')
    img_path = sorted(img_path)
    print(img_path[10])

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    color_mean = (0.5, 0.5, 0.5)
    color_std = (0.5, 0.5, 0.5)
    width = 448
    height = 448

    use_pretrained=False
    model = models.resnet50(pretrained=use_pretrained)
    model.fc = nn.Linear(in_features=2048, out_features=4)

    state_dict = torch.load('./weights/resnet50_150.pth', map_location={'cuda:0':'cpu'})
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)
    model.eval()
    logs = []

    for i in range(len(img_path)):
        name = img_path[i]
        img_name = name.split('/')[3]

        img = Image.open(name)
        img = img.resize((width, height))
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, color_mean, color_std)
        img = img.unsqueeze(0)
        img = img.to(device)

        output = model(img)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.item()
        #predicted = predicted.detach().numpy()[0].to('cpu')
        '''
        output = output.to('cpu')
        output = output.detach().numpy()[0]
        output = output.argsort()[::-1]
        predicted = output[0]
        '''
        #print(predicted)
        result = {'img_name':img_name, 'prediction_label':predicted}
        logs.append(result)
    
    df = pd.DataFrame(logs)
    df.to_csv('./submit/submit_csv_resnet50_150.csv', index=False, header=False)


if __name__ =='__main__':
    main()
    print('finish')