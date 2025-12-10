import torch
from sklearn.datasets import make_moons
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

writer = SummaryWriter("loss1")
def get_moons_dataset(n_samples=1000, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,1)
    return X, y


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)

    def forward(self,x):
        x = self.linear(x)
        x = 1/(1+torch.exp(-x))
        return x

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        L = -y*torch.log(x)-(1-y)*torch.log(1-x)
        return L


class Opt:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
def train():
    X,y = get_moons_dataset()
    net = Model()
    opt = Opt(net.parameters(),lr=0.1)
    loss_f =Loss()
    epochs =100
    dataset = data.TensorDataset(X,y)
    dataloader = data.DataLoader(dataset,batch_size=30,shuffle=True)
    for epoch in range(epochs):
        loss = 0
        pbar = tqdm(dataloader)
        for features,labels in pbar:
            opt.zero_grad()
            out = net(features)
            l = loss_f(out,labels)
            l = l.mean()
            l.backward()
            opt.step()
            loss += l.item()
        print("loss: ",loss)
        writer.add_scalar("loss",loss,epoch+1)

    # 1. 数据散点
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap="viridis", s=10)

    # 2. 提取参数
    W = net.linear.weight.detach().cpu().numpy()[0]
    b = net.linear.bias.detach().cpu().numpy()

    # 3. 决策边界
    x_vals = np.linspace(X[:,0].min()-0.2, X[:,0].max()+0.2, 200)
    y_vals = -(W[0] * x_vals + b) / W[1]

    plt.plot(x_vals, y_vals, 'r', label="Decision Boundary")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.show()

if __name__ == '__main__':
    train()







