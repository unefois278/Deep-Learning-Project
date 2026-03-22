import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.datasets import make_regression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# 造线性回归的数据
def create_dataset():
    x, y, coef = make_regression(
        n_samples=100,
        n_features=1,
        noise=10,
        coef=True,
        bias=14.5,
        random_state=114514
    )
    print(x, y, coef) #但是类型是ndarray

    #封装成tensor再返回
    x = torch.tensor(x, dtype=torch.float) #特征
    y = torch.tensor(y, dtype=torch.float) #标签
    return x, y, coef


#训练模型
def train(x, y, coef):
    #tensor -> dataset -> dataloader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True) #每轮100条数据，每批20条，共5批

    model = nn.Linear(1, 1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #具体的训练过程

    epochs = 100
    loss_list = []
    total_loss = 0.0
    total_sample = 0

    for epoch in range(epochs):
        for train_x, train_y in dataloader:
            y_predict = model(train_x)
            loss = criterion(y_predict, train_y.reshape(-1, 1)) #n行1列

            total_loss += loss.item()
            total_sample += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(total_loss / total_sample)
        print(f"轮数：{epoch+1}, 平均损失值：{total_loss / total_sample}")

    print(f"{epochs}轮的平均损失分别为{loss_list}")
    print(f"模型参数，权重：{model.weight}，偏置：{model.bias}")



    #损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title("损失值曲线变化图")
    plt.grid()
    plt.show()

    #预测值和真实值的关系与曲线
    plt.scatter(x, y)
    y_predict = torch.tensor(data = [v * model.weight + model.bias for v in x])
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x])
    plt.plot(x, y_predict, color="red", label="预测值")
    plt.plot(x, y_true, color="green", label="真实值")
    plt.legend()
    plt.grid()
    plt.show()




if __name__ == '__main__':
    x, y, coef = create_dataset()
    train(x, y, coef)
