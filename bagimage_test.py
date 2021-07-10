# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:23:50 2020

@author: yileb
"""
import bagnets.pytorchnet
model = bagnets.pytorchnet.bagnet33(pretrained=True).cuda()

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import time
import numpy as np
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def get_paths_labels():
    # 数据集地址
    imagenet_path = r"E:\classification\res_image"
    photo_path = r"E:\classification\res_image\ILSVRC2012_img_val"

    file_names = os.listdir(photo_path)
    image_paths = []
    for file_name in file_names:
        full_path = os.path.join(photo_path, file_name)
        image_paths.append(full_path)
    # 从val_caffe label中读取对应的imagenet_numlabel(caffe)
    with open(os.path.join(imagenet_path, 'val_caffe label.txt'), 'r') as file3:
        names = []
        img_numlabels = []
        for line in file3.readlines():
            name, img_numlabel = line.strip().split()
            names.append(name)
            img_numlabels.append(img_numlabel)
    image_labels = img_numlabels

    return image_paths, image_labels

#得验证集图像地址和标签
image_paths, image_labels = get_paths_labels()

# 记录top1、top5正确数目
top1_cnt = 0
top5_cnt = 0

# 用于记录测试用时
start = time.time()

# 记录总图像数目
cnt = 0

#预处理
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    #转为tensor并归一化到[0，1]
    transforms.ToTensor(),
    #用均值和标准差处理tensor，[-1,1]
    transforms.Normalize(mean = mean, std = std),
    ])

for image_path, image_label in zip(image_paths, image_labels):
    img = Image.open(image_path)

    # 若遇到灰度图像，转化为RGB以适应normalize函数
    if img.mode != 'RGB' :
       img = img.convert('RGB')

    x = transform(img)
    # 将tensor迁移到显存中
    x=x.cuda()
    # 添加一个维度 [1,3,224,224]
    x=x.unsqueeze(0)

    with torch.no_grad():

        # 切换到评估模式
        model.eval()
        res = model(x)
        top5 = res.topk(5,1,True,True)

        if int(image_label) in top5.indices:
           top5_cnt += 1

        if int(image_label) == top5.indices[0][0]:
           top1_cnt += 1

        cnt += 1

        # 每10000张图片，输出一次测试耗时
        if cnt % 10000 == 0 :
            end = time.time()
            print('%d steps: %f' % (cnt, end - start))
            start = end

print('top1 accuracy:', top1_cnt / 50000)
print('top5 accuracy:', top5_cnt / 50000)

