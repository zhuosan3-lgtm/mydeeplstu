import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils import data
from torch import nn
from tqdm import tqdm


writer = SummaryWriter('loss3')
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转为张量（0-1范围）
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差（官方推荐）
])

# 2. 加载训练集和测试集（root为数据保存路径，download=True自动下载）
train_dataset = datasets.MNIST(
    root='./data',        # 数据保存到当前目录的data文件夹
    train=True,           # 训练集
    download=True,        # 首次运行自动下载
    transform=transform   # 应用预处理
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,          # 测试集
    download=True,
    transform=transform
)

train_loader = data.DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=0)
for images, labels in train_loader:
    print(f"批次图像形状: {images.shape}")  # (64, 1, 28, 28) → (批量数, 通道数, 高度, 宽度)
    print(f"批次标签形状: {labels}")  # (64,)
    print(f"第一个样本标签: {labels[0].item()}")
    break

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 =nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),nn.Linear(64*7*7,128),nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        return self.n1(x)

net = model().to('cuda')
loss_f = nn.CrossEntropyLoss().to('cuda')
opt = torch.optim.Adam(net.parameters(),lr = 0.001)
for epoch in range(20):
    loss = 0
    t = 0
    ta = 0
    pbar = tqdm(train_loader)
    for features,labels in pbar:
        features = features.to('cuda')
        labels = labels.to('cuda')
        opt.zero_grad()
        out = net(features)
        l = loss_f(out,labels)
        l.backward()
        opt.step()
        loss +=l.item()
        y_hat = out.argmax(axis=1)
        cmp = y_hat==labels
        acc = float(cmp.type(labels.dtype).sum())
        ta += acc
        t += len(labels)

    print("loss: ",loss)
    writer.add_scalar("loss",loss,epoch+1)
    ac = ta/t
    print('acc:',ac)


