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
writer = SummaryWriter(r"C:\Users\20243\Desktop\github学习项目\linergression\loss")
for epoch in range(epochs):
    total_loss = 0.0
    batch_num = 0
    pbar = tqdm(data_iter, desc=f"Epoch {epoch+1}/{epochs}")
    for features, labels in pbar:
        opt.zero_grad()
        pred = net(features)
        loss = loss_f(pred, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        batch_num += 1
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    # 关键：计算平均损失，确保是标量，且每个epoch写一次
    avg_loss = total_loss / batch_num
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    # 必须确保这里的avg_loss是Python标量（.item()或float类型）
    writer.add_scalar('Training_Loss', avg_loss, global_step=epoch+1)

writer.flush()
writer.close()  # 必须关闭，确保数据写入
for p in net.parameters():
    print(p)
