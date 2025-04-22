# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time
from utils import progress_bar
from models.cltr.final_model import F_Model
from models.vit.plain_vit import Plain_Vit
from models.resnet.resnet import ResNet18
from models.resnet.resnet import ResNet50
from models.mobilenet.MobileNet import MobileNet
from models.senet.SENet import SENet18
from models.densenet.DenseNet import DenseNet_cifar
from models.vgg.VGG import VGG
from models.shufflenet.ShuffleNet import ShuffleNetG3
from models.efficientnet.EfficientNet import EfficientNetB0


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--net', default='ictr')
parser.add_argument('--bs', default='64')
parser.add_argument('--size', default="32")

args = parser.parse_args()
bs = int(args.bs)
imsize = int(args.size)
model_name=args.net
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')

size = 224

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)


if model_name == 'ictr':
    backbone_params=dict(
        model_name="vit_base_patch16_224", 
        num_classes=0,
        pretrained=True
    )

    neck_params = dict(
        in_dim = 192, 
        out_dims = [128, 256, 512, 1024],
    )

    head_params = dict(
        in_channels=[128, 256, 512, 1024], 
        num_classes=100,
        hidden_dim=256,
        num_queries=50,    
        nheads=8,
        dim_feedforward=1024,
        dec_layers=4,  
        pre_norm=True
    )


    net = F_Model(
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
    )

elif model_name == 'vit_plain':
    backbone_params=dict(
        model_name="vit_base_patch16_224", 
        pretrained=True
    )
    head_params=dict(
        input_channel=1000,
        num_class=100,
    )
    net = Plain_Vit(
        backbone_params=backbone_params,
        head_params=head_params,
    )

elif model_name=='resnet18':
    net = ResNet18()
    
elif model_name=='resnet50':
    net = ResNet50()
elif model_name=='convnext':
    net = convnext_base()
elif model_name=='mobilenet':
    net = MobileNet()
elif model_name=='senet':
    net = SENet18()
elif model_name=='densenet':
    net = DenseNet_cifar()
elif model_name=='vgg':
    net = VGG('VGG19')
elif model_name=='shufflenet':
    net = ShuffleNetG3()
elif model_name=='efficientnet':
    net=EfficientNetB0()
    
    
def test():
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_100/'+model_name+'/'+model_name+'_new_pretrained224-ckpt.pth')
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    net.eval()
    top_1_correct = 0
    top_5_correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            # Top-1 准确率
            _, predicted = outputs.max(1)
            top_1_correct += predicted.eq(targets).sum().item()

            # Top-5 准确率
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct_top5 = top5_pred.eq(targets.view(1, -1).expand_as(top5_pred))
            top_5_correct += correct_top5[:5].sum().item()

            # 收集所有真实标签和预测结果
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

        total = len(testloader.dataset)
        top_1_acc = 100. * top_1_correct / total
        top_5_acc = 100. * top_5_correct / total

        print(f"Top-1 Accuracy: {top_1_acc:.2f}%")
        print(f"Top-5 Accuracy: {top_5_acc:.2f}%")

        # 计算 F1 Score
        f1 = f1_score(all_targets, all_predictions, average='macro')
        print(f"F1 Score (Macro): {f1:.4f}")
        
        
        def calculate_map(probabilities, targets, num_classes):
            aps = []
            for cls in range(num_classes):
                # 提取当前类别的预测概率和真实标签
                cls_probs = np.array([prob[cls] for prob in probabilities])
                cls_targets = np.array([1 if t == cls else 0 for t in targets])

                # 计算 Precision-Recall 曲线
                precision, recall, _ = precision_recall_curve(cls_targets, cls_probs)
                ap = auc(recall, precision)
                aps.append(ap)

            return np.mean(aps)

        num_classes = outputs.shape[1]  # 假设输出类别数为 outputs 的第二维大小
        map_value = calculate_map(all_probabilities, all_targets, num_classes)
        print(f"mAP: {map_value:.4f}")
        
test()
    
    
    
    