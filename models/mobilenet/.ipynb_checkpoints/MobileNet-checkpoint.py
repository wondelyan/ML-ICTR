import torch
from torch import nn, optim
from torch.nn import functional as F


#  深度可分离卷积 DSC, 深度卷积 Depthwise + 逐点卷积 Pointwise
class DSCconv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DSCconv, self).__init__()
        self.depthConv = nn.Sequential(  # 深度卷积, (DW+BN+ReLU)
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                      padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True))
        self.pointConv = nn.Sequential(  # 逐点卷积, (PW+BN+ReLU)
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        return x


class MobileNet(nn.Module):
    cfg = [(64, 1),  # (in=32, out=64, s=1)
           (128, 2),  # (in=64, out=128, s=2)
           (128, 1),  # (in=128, out=128, s=1)
           (256, 2),  # (in=128, out=256, s=2)
           (256, 1),  # (in=256, out=256, s=1)
           (512, 2),  # (in=256, out=512, s=2)
           (512, 1),  # (in=512, out=512, s=1)
           (512, 1),  # (in=512, out=512, s=1)
           (512, 1),  # (in=512, out=512, s=1)
           (512, 1),  # (in=512, out=512, s=1)
           (512, 1),  # (in=512, out=512, s=1)
           (1024, 2),  # (in=512, out=1024, s=2)
           (1024, 1)]  # (in=1024, out=1024, s=1)

    def __init__(self, num_classes=100):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_ch=32)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # torch.Size([batch, 1024, 1, 1])
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_ch):
        layers = []
        for x in self.cfg:
            out_ch = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(DSCconv(in_ch, out_ch, stride))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
