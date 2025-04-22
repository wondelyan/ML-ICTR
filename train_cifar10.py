# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

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

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', default='ictr')
parser.add_argument('--bs', default='16')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='100')

args = parser.parse_args()

bs = int(args.bs)
imsize = int(args.size)

model_name=args.net
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# if model_name != 'ictr':
#     size = imsize
# elif model_name=='ictr' or model_name=='vit_plain':
#     size = 384

size = 224
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Prepare dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')

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
        num_classes=10,
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
        num_class=10,
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



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+model_name+'/cltr_pretrained224_100-150-ckpt.pth')
    net.load_state_dict(checkpoint['model'])
#     best_acc = checkpoint['accuracy']
#     start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(net.parameters(), lr=args.lr)
 
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=False)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=False):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+model_name+'/'+model_name+'_new_pretrained224-ckpt.pth')
#         torch.save(state, 'resnet50_pretrained32-ckpt.pth')
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_'+model_name+'_224/new_log_'+model_name+'_224.txt', 'a') as appender:
#     with open(f'log_resnet50_32.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

    
net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)

    # Write out csv..
    with open(f'log/log_'+model_name+'_224/new_log_'+model_name+'_224.csv', 'w') as f:
#     with open(f'log_resnet50_32.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)


    
