from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
def get_moons_dataset(n_samples=1000, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,1)
    return X, y
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Sequential(
            nn.Linear(2,16),nn.ReLU(),
            nn.Linear(16,4),nn.ReLU(),
            nn.Linear(4,1),nn.Sigmoid()
        )

    def forward(self,x):
        x = self.n1(x)
        return x
def train():
    device = 'cpu'
    net = Model().to(device)
    loss_f = nn.BCELoss().to(device)
    opt = torch.optim.Adam(net.parameters(),lr=0.05)
    x,y = get_moons_dataset()
    dataset = data.TensorDataset(x,y)
    dataloader = data.DataLoader(dataset,batch_size=30,shuffle=True)
    for epoch in range(20):
        loss = 0
        pbar = tqdm(dataloader)
        for features,labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            out = net(features)
            l = loss_f(out,labels)
            l.backward()
            opt.step()
            loss += l.item()
        print('loss:',loss)


    # mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 300),
        np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 300)
    )

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        logits = net(grid).reshape(xx.shape)
        probs = torch.sigmoid(logits)

    plt.contourf(xx, yy, probs, levels=50, cmap='RdBu', alpha=0.6)
    plt.scatter(x[:, 0], x[:, 1], c=y.squeeze(), cmap='bwr')
    plt.title("MLP Nonlinear Decision Boundary")
    plt.show()


if __name__ == '__main__':
    train()




