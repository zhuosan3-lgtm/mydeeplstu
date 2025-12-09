import argparse

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils import data

def parse_args():
    parser = argparse.ArgumentParser(description='逻辑回归模型训练参数')

    parser.add_argument('--epoches',type=int,default=200)
    parser.add_argument('--lr',default=0.05)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--batch_size', type=int, default=30)

    return parser.parse_args()

args = parse_args()
writer = SummaryWriter("loss2")
def get_moons_dataset(n_samples=1000, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,1)
    return X, y

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner = nn.Linear(2,1)
    def forward(self,x):
        x = self.liner(x)
        y = 1/(1+torch.exp(-x))
        return y

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        return -(y*torch.log(x)+(1-y)*torch.log(1-x))

class Optim(nn.Module):
    def __init__(self,params,lr):
        super().__init__()
        self.lr = lr
        self.params = list(params)
    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param -= self.lr*param.grad
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
def train():
    net = Model().to(args.device)
    loss_f = BCELoss().to(args.device)
    opt = Optim(net.parameters(),lr=args.lr)
    X,y = get_moons_dataset()
    dataset = data.TensorDataset(X,y)
    dataloader = data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    for epoch in  range(args.epoches):
        loss = 0
        pbar = tqdm(dataloader)
        for features,labels in pbar:
            opt.zero_grad()
            features = features.to(args.device)
            labels = labels.to(args.device)
            out = net(features)
            l = loss_f(out,labels)
            l = l.mean()
            l.backward()
            opt.step()
            loss += l.item()
        print('loss:',loss)
        writer.add_scalar('loss:',loss,epoch+1)
#    torch.save(net,"end.pth")
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="viridis", s=10)

    # 2. 提取参数
    W = net.liner.weight.detach().cpu().numpy()[0]
    b = net.liner.bias.detach().cpu().numpy()

    # 3. 决策边界
    x_vals = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 200)
    y_vals = -(W[0] * x_vals + b) / W[1]

    plt.plot(x_vals, y_vals, 'r', label="Decision Boundary")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.show()
if __name__ == '__main__':
    train()



































