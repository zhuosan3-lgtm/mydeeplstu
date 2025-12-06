import torch
import argparse
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='线性回归模型训练参数')

    # 数据参数
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num',type=int,default=1000)
    parser.add_argument('--true_w',default=torch.tensor([4,-8.1]))
    parser.add_argument('--true_b',default=3.7)
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.03, help='学习率')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device',type=str,default="cuda")
    return parser.parse_args()


# 使用
args = parse_args()
writer = SummaryWriter("loss")
def synthetic_data(w,b,num):
    "生成y=Xw+b的数据"
    x = torch.normal(0,1,(num,len(w)))
    y = torch.matmul(x,w)+b
    y += torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))
class line_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(2,1)
        )
    def forward(self,x):
        return self.l1(x)
def dataloader(num):
    features,labels = synthetic_data(args.true_w,args.true_b,num)
    dataset = data.TensorDataset(features,labels)
    dataloader = data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    return dataloader

def train():
    data_train = dataloader(args.num)
    data_val = dataloader(int(args.num/2))
    net = line_model().to(args.device)
    loss_f = nn.MSELoss().to(args.device)
    opt = torch.optim.Adam(net.parameters(),lr=args.lr)
    epochs = args.epochs

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        net.train()
        pbar = tqdm(data_train)
        for features,labels in pbar:
            features = features.to(args.device)
            labels = labels.to(args.device)
            opt.zero_grad()
            out = net(features)
            l = loss_f(out,labels)
            l.backward()
            opt.step()
            train_loss += l.item()
        writer.add_scalar("train_loss",train_loss,epoch+1)
        print('train_loss: ',train_loss)
        net.eval()
        with torch.no_grad():
            for features,labels in data_val:
                features = features.to(args.device)
                labels = labels.to(args.device)
                out = net(features)
                l = loss_f(out,labels)
                val_loss += l.item()
        writer.add_scalar("val_loss", val_loss, epoch + 1)
        print('val_loss: ',val_loss)
    for p in net.parameters():
        print(p)

if __name__ == '__main__':
    train()
