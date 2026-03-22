#0.导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary
import os

BATCH_SIZE = 256

#1.准备CIFAR10数据集
def create_dataset():
    #初始化训练集与测试集
    train_dataset = CIFAR10(root = './', train = True, transform = ToTensor(), download = True )
    test_dataset = CIFAR10(root = './', train = False, transform = ToTensor(), download = True )
    return train_dataset, test_dataset





#2.搭建卷积神经网络
class ImageModel(nn.Module):
    #(1)初始化
    def __init__(self):
        super().__init__()

        #第1个卷积层(输入3通道 输出6通道 卷积核大小3*3 步长1 填充0)
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        #第1个池化层(池化窗口大小2*2 步长2 填充0)
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        #第2个卷积层与池化层
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        #全连接层
        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    #(2)前向传播
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        
        #将数据进行拉平 方便后续进行全连接层的处理
        x = x.reshape(x.size(0), -1)

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.output(x)
    




#3.训练图像分类模型
def train(train_dataset, device):
    dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    model = ImageModel().to(device)
    criterion = nn.CrossEntropyLoss() #多分类交叉熵函数 自带softmax的功能
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)


    epochs = 50
    for epoch_idx in range(epochs):
        model.train()
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_correct += (torch.argmax(y_pred, dim = -1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
        
        print(f"epoch:{epoch_idx + 1}, loss:{total_loss / total_samples:.5f}, accuracy:{total_correct / total_samples:.2%}, time:{time.time() - start:.2f}s")
    
    os.makedirs("./model", exist_ok = True)
    torch.save(model.state_dict(), "./model/image_model.pth")





#4.测试模型
def evaluate(test_dataset, device):
    dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)
    
    model = ImageModel().to(device)
    model.load_state_dict(torch.load("./model/image_model.pth"))

    model.eval()

    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            #这里要用argmax模拟softmax的功能，因为测试集没有经过crossentropyloss
            y_pred = torch.argmax(y_pred, dim = -1)

            total_correct += (y_pred == y).sum().item()
            total_samples += len(y)
    

    print(f"accuracy:{total_correct / total_samples:.2%}")






if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是: {device}")

    train(train_dataset, device)
    
    evaluate(test_dataset, device)