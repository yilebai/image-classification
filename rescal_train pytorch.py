# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:15:37 2020

@author: yileb
"""
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import time
import sys
from sklearn.model_selection import train_test_split
import progressbar
import copy

data_dir = '/content/drive/My Drive/test/101_ObjectCategories/'
#data_dir = '/classification/res_caltech/101_ObjectCategories/'
batch_size = 128
num_epochs = 40
classnum = 102
# feature extracting标志 为真时只调参最后一层，为假时训练所有 
feature_extract = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#划分0.5训练集0.25验证集0.25测试集，已运行过划分好
def split():
    categories = os.listdir(data_dir)  # 102 categories
    bar = progressbar.ProgressBar()
    for i in bar(range(len(categories))):

        category = categories[i]  
        cat_dir = os.path.join(data_dir, category)
    
        images = os.listdir(cat_dir)  
    
        images, images_test = train_test_split(images, test_size=0.25)
        images_train, images_val = train_test_split(images, test_size=0.33)
        image_sets = images_train, images_test, images_val
        labels = 'train', 'test', 'val'
    
        # 移动至相应文件夹
        for image_set, label in zip(image_sets, labels):
            dst_folder = os.path.join(data_dir, label, category)  # create folder
            os.makedirs(dst_folder)
            for image in image_set:
                #重命名 ../cat/xxx.jpg to ../label/cat/xxx.jpg
                src_dir = os.path.join(cat_dir, image)
                dst_dir = os.path.join(dst_folder, image)
                os.rename(src_dir, dst_dir)  
                
        # 移除空文件夹  
        os.rmdir(cat_dir)  

#split()


#图像预处理
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

image_transforms = {
    # 训练集数据增强
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]),
    # 验证集不用数据增强
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]),
}
 
# 加载数据集
data = {
    'train':
    datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=image_transforms['val']),
}
 
# 迭代，允许打乱顺序
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True)
}

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#预训练模型重塑
def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    
    model_ft = None

    if model_name == "resnet50":

        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16":

        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
                                         nn.Linear(num_ftrs, 256),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.4),
                                         nn.Linear(256, num_classes),
                                               )
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

#预训练模型
model_ft = initialize_model('resnet50', classnum, feature_extract=True, use_pretrained=True)
model_ft = model_ft.to(device)

#为预训练模型创建优化器
params_to_update = model_ft.parameters()
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        #只更新true的参数，即刚刚调整过的参数
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
#其实可以取消
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

#optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(params_to_update)

#无训练模型
model = initialize_model('resnet50', classnum, feature_extract=False, use_pretrained=False)
model = model.to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
optimizer = optim.Adam(model.parameters())

#模型训练及验证
def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    since = time.time()

    val_acc_history = []
    val_loss_history = []


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有train和evaluate阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            # 迭代
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                #inputs = inputs.cuda()
                #labels = labels.cuda()

                # parameter gradients置零
                optimizer.zero_grad()

                # forward
                # 只在训练阶段track history
                with torch.set_grad_enabled(phase == 'train'):
                    # 得到模型输出，计算loss
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + 只在训练阶段优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最好模型
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history


#损失函数
criterion = nn.CrossEntropyLoss()
   
# Train and evaluate
model, hist, hist1 = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, is_inception = False)
torch.save(model.state_dict(), 'res.pth')
model_ft, hist_ft, hist_ft1 = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs, is_inception = False)
torch.save(model_ft.state_dict(), 'res_ft.pth')

ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in hist_ft]

plt.figure()
plt.title("Validation Accuracy with Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Untrained")
plt.plot(range(1,num_epochs+1),shist,label="Pretrained")
plt.ylim((0,1.))
plt.xticks(np.arange(0, num_epochs+1, 5.0))
plt.legend()
plt.show()

ohist1 = []
shist1 = []

for h in hist1:
    ohist1.append(h)
    
for h in hist_ft1:
    shist1.append(h)


plt.figure()
plt.title("Validation Loss with Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Loss")
plt.plot(range(1,num_epochs+1),ohist1,label="Untrained")
plt.plot(range(1,num_epochs+1),shist1,label="Pretrained")
plt.xticks(np.arange(0, num_epochs+1, 5.0))
plt.legend()
plt.show()