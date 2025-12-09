import os
import shutil
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse

def parse():
    parse = argparse.ArgumentParser("CNN分类MINST参数设置")
    parse.add_argument('--batch_size',type=int,default=32)
    parse.add_argument('--epoches',type=int,default=20)
    parse.add_argument('--device',type=str,default='cuda')
    parse.add_argument('--loss_f',default=nn.CrossEntropyLoss)
    parse.add_argument('--optim',default=torch.optim.Adam)
    parse.add_argument('--lr',default=0.001)
    parse.add_argument('--log_dir',type=str,default='loss4')
    return parse.parse_args()


args = parse()
writer = SummaryWriter(args.log_dir)
def dataloader():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.1307,0.3081)])
    train_set = datasets.MNIST(
        './Minst',
        train=True,
        transform=transform,
        download=True
    )
    test_set =datasets.MNIST(
        './Minst',
        train=False,
        transform=transform,
        download=True
    )
    train_loader = data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)
    return train_loader,test_loader
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Sequential(nn.Conv2d(1,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(2),nn.Flatten(),nn.Linear(64*7*7,1024),nn.Dropout(0.2),nn.ReLU(),
                                nn.Linear(1024,10)
        )
    def forward(self,x):
        return self.n1(x)
def acc(y_h,y):
    h = y_h.argmax(axis=1)
    cmp = h==y
    return float(cmp.sum())


def delete_checkpoint(checkpoint_path):
    """
    删除检查点（兼容文件/文件夹）
    :param checkpoint_path: 检查点路径（文件或文件夹）
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return  # 路径不存在，无需删除

    if os.path.isfile(checkpoint_path):
        # 删除单个文件（如 .pth/.ckpt/.h5）
        os.remove(checkpoint_path)
        print(f"已删除旧检查点文件: {checkpoint_path}")
def train():
    net = Model().to(args.device)
    loss_f = args.loss_f().to(args.device)
    opt = args.optim(net.parameters(),lr = args.lr)
    train,test = dataloader()
    best_ac = 0
    for epoch in range(args.epoches):
        pbar1 = tqdm(train)
        train_loss = 0
        train_acc = 0
        train_num = 0
        net.train()
        pbar2 = tqdm(train)
        test_loss = 0
        test_acc = 0
        test_num = 0
        net.train()
        for features,labels in pbar1:
            features = features.to(args.device)
            labels = labels.to(args.device)
            opt.zero_grad()
            out = net(features)
            l = loss_f(out,labels)
            l.backward()
            opt.step()
            train_loss += l.item()
            t_a = acc(out,labels)
            train_acc += t_a
            train_num += len(labels)
        print('epoch: ',epoch+1,' train_loss: ',train_loss)
        train_ac = train_acc/train_num
        print('epoch: ', epoch + 1, ' train_acc: ', train_ac)
        writer.add_scalar('train_loss',train_loss,epoch+1)
        writer.add_scalar('train_acc', train_ac, epoch + 1)



        net.eval()
        for features,labels in pbar2:
            features = features.to(args.device)
            labels = labels.to(args.device)
            out = net(features)
            l = loss_f(out,labels)
            test_loss += l.item()
            t_a = acc(out,labels)
            test_acc += t_a
            test_num += len(labels)
        print('epoch: ',epoch+1,' test_loss: ',test_loss)
        test_ac = test_acc/test_num
        print('epoch: ', epoch + 1, ' test_ac: ', test_ac)
        writer.add_scalar('test_loss',test_loss,epoch+1)
        writer.add_scalar('test_acc', test_ac, epoch + 1)
        if test_ac>best_ac:
            best_ac= test_ac
            delete_checkpoint("CNN_best.pth")
            torch.save(net,"CNN_best.pth")


def classify_mnist_test_sample(sample_idx):
    """
    分类MNIST测试集中指定索引的单张图片
    :param sample_idx: 测试集索引（0-9999）
    :return: 真实数字、预测数字、预测概率
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.1307,0.3081)])
    # 加载MNIST测试集（无需下载，已存在）
    test_set =datasets.MNIST(
        './Minst',
        train=False,
        transform=transform,
        download=True
    )
    # 获取指定样本
    img_tensor, true_digit = test_set[sample_idx]
    img_tensor = img_tensor.unsqueeze(0).to('cuda')  # 增加batch维度
    model = torch.load("CNN_best.pth")
    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred_digit = torch.argmax(prob, dim=1).item()
    # 输出结果
    print(f"真实数字：{true_digit}")
    print(f"预测数字：{pred_digit}")

# 执行分类（比如选测试集第100张图片）
if __name__ == '__main__':
    train()
    classify_mnist_test_sample(sample_idx=1013)



        