import torch
from torch.utils import data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def synthetic_data(w,b,num_example):
    """生成 y=Xw+b 的噪声"""
    X = torch.normal(0,1,(num_example,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X , y.reshape((-1,1))

ture_w = torch.tensor([4,-8.1])
ture_b = 3.7
features,labels = synthetic_data(ture_w,ture_b,1000)
dataset = data.TensorDataset(features,labels)
#print(dataset[0])
data_iter = data.DataLoader(dataset,batch_size=10,shuffle=True)
#print(next(iter(data_iter)))

class line_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.line = nn.Sequential(
            nn.Linear(2,1)
        )

    def forward(self,x):
        return self.line(x)
net = line_net()
lr = 0.03
epochs = 50
loss_f = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(),lr)
writer = SummaryWriter("lo")
device = torch.device("cuda"if torch.cuda.is_available()else "cpu" )
net.to(device)
loss_f.to(device)


for epoch in range(epochs):
    l = 0
    pbar = tqdm(data_iter)  #更新一遍才会有新的进度条，放在循环外面就只有一个进度条
    for features,labels in pbar:
        features = features.to(device)#要赋值才会得到cuda
        labels = labels.to(device)
        opt.zero_grad()
        x = net(features)
        loss = loss_f(x,labels)
        loss.backward()
        opt.step()
        l +=loss
    l = l.item()
    print(l)
    writer.add_scalar('los',l,epoch+1)
writer.flush()
writer.close()
for p in net.parameters():
    print(p.data.cpu())














