# -*- encoding: utf-8 -*-
"""
File neural_net.py
Created on 2024/1/20 18:55
Copyright (c) 2024/1/20
@author: 
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设计模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm2d(20),
        )
        # 输出10个类别
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=320, out_features=10)
        )

    def forward(self, x):
        # x: B C=10 H=12 W=12
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        return x

def construct_data_loader(batch_size):
    # 数据的归一化
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # 训练集
    train_dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 测试集
    test_dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(train_loader):
    for (images, target) in train_loader:
        # images shape: B C=1 H W
        outputs = model(images)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_model(test_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for (images, target) in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('[%d / %d]: %.2f %% ' % (i + 1, epoch, 100 * correct / total))

if __name__ == '__main__':
    # 定义超参数
    # 批处理大小
    batch_size = 128
    # 学习率
    lr = 0.002
    # 动量
    momentum = 0.5
    # 训练的epoch数
    epoch = 10
    # 构建模型
    model = Net()
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_loader, test_loader = construct_data_loader(batch_size)
    for i in range(epoch):
        # 训练
        train_model(train_loader)
        # 测试
        test_model(test_loader)