import numpy as np
from PIL import Image
import cv2
import random
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torchvision.datasets import ImageFolder
import pandas as pd
import torch.optim as optim


def train(model, num_epochs,train_loader, val_loader):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    weights_path = '/kw_resources/signate/signate_lemon_classification/weights/resnet_'+str(num_epochs)+'.pth'
    log_path = '/kw_resources/signate/signate_lemon_classification/log_out.csv'
    #weights_path = './weights/efficientnetb4_'+str(num_epochs)+'.pth'
    #log_path = './log_out.csv'
    for epoch in range(1, num_epochs+1):
        if epoch == 1:
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                logs = checkpoint['logs']
                if epoch == num_epochs:
                    break
    model.to(device)
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    logs = []

    for epoch in range(1, num_epochs+1):
        correct = 0 #正解したデータの総数
        total = 0 #予測したデータの総数
        running_loss = 0
        model.train()
        for data in train_loader:
            img, label = data[0], data[1]
            
            img , label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            _,predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            # 予測したデータ数を加算
            correct += (predicted == label).sum().item()
            #correct += torch.sum(predicted==v_label.data)
        train_loss = running_loss/len(train_loader)
        train_acc=correct/total
        
        if epoch % 1 == 0:
            correct = 0 #正解したデータの総数
            total = 0 #予測したデータの総数
            running_loss = 0
            model.eval()
            for data in val_loader:
                img, label = data[0], data[1]
                
                img , label = img.to(device), label.to(device)
                outputs = model(img)
                loss = criterion(outputs, label)
                running_loss += loss.item()
                _,predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                # 予測したデータ数を加算
                correct += (predicted == label).sum().item()
                #correct += torch.sum(predicted==v_label.data)
            print('Epoch {}/{}'.format(epoch,num_epochs))
            
            val_loss = running_loss/(len(val_loader))
            val_acc = correct/total
            
            

            print('epoch : {},  train_loss : {}, train_acc : {}, val_loss : {},val_acc : {}'.format(epoch, train_loss, train_acc, val_loss, val_acc))

            #ログを保存
            log_epoch = {'epoch' : epoch, 'train_loss' : train_loss,'train_acc':train_acc, 'val_loss' : val_loss, 'val_acc':val_acc }
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv(log_path)
            #df.to_csv('log_out.csv')
        
        if epoch % 5 == 0:
            print('---------------------------------------------------------------')
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.module.state_dict(),
                'logs':logs
            }, weights_path)
            #torch.save(model.module.state_dict(),weights_path)
            #torch.save(model.module.state_dict(),'./weights/resnet_'+str(epoch)+'.pth')
        
    
    return model