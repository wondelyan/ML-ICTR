# 深度学习模型在CIFAR-10、CIFAR-100数据集的实践
本实验中包含的模型有DenseNet，EfficientNet、MobileNet、ResNet18、ResNet50、SENet、ShuffleNet、VGG19、ViT-plain以及独自创新的ML-ICTR模型。

## 环境设置
训练与测试环境为: Python3.8, PyTorch 2.0.1, Ubuntu 20.4, CUDA 11.8. 运行以下命令安装相应的Python包。
```
pip3 install -r requirements.txt
```
# 文件说明
checkpoint存放训练完成的模型权重文件，data存放CIFAR数据集，log存放训练过程记录，models存放模型代码，pretrained存放预训练代码

# 训练
`python train_cifar10.py --lr *** --net *** --bs *** --size *** --n_epoch ***`  该过程会自动下载CIFAR10数据集，存放到data文件下

`python train_cifar100.py  --lr *** --net *** --bs *** --size *** --n_epoch ***`  lr、bs、size、n_epoch都可以缺省，赋予默认值

# 测试
`python test_cifar10.py --net *** --bs *** --size ***`

`python test_cifar100.py --net *** --bs *** --size ***`  将会输出Top-1、Top-5、f1 score、mAP指标值

# 绘制折线图
`python draw_pic.py`  根据log文件中的记录，绘制折线图

# 恢复CIFAR-10原始RGB图像
`python to_rgb.py`

# 模型权重下载
链接: https://pan.baidu.com/s/1AZwDy4RCO_lnFCUmnNJ7ug 提取码: 82yi


