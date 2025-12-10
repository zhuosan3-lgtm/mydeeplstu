import os
from torchvision.utils import save_image
from torch import nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse
from torch.utils import data
from torch.nn import functional as F
def Parse():
    parse = argparse.ArgumentParser("VAE模型参数设置")
    parse.add_argument('--batch_size',type=int,default=64)
    parse.add_argument('--lr',default=0.001)
    parse.add_argument('--epoches',type=int,default=10)
    parse.add_argument('--device',type=str,default='cuda')
    parse.add_argument('--optim',default=torch.optim.Adam)
    parse.add_argument('--a_kl',default=1)
    parse.add_argument('--num_sam',type=int,default=32)
    return parse.parse_args()

args = Parse()
writer = SummaryWriter('loss')

def dataloader():
    transform = transforms.Compose([transforms.ToTensor()])#,transforms.Normalize(0.1307,0.3081)])
    train_set = datasets.MNIST('./data',train=True,transform=transform,download=True)
    test_set = datasets.MNIST('./data',train=False,transform=transform,download=True)
    train_loader = data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)
    return train_loader,test_loader
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(1,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(2),nn.Flatten(),nn.Linear(64*7*7,1024),nn.ReLU()
        )
        self.lvn = nn.Linear(1024,128)
        self.un = nn.Linear(1024,128)
        self.decode = nn.Sequential(nn.Linear(128,1024),nn.ReLU(),
                                    nn.Linear(1024,out_features=64*7*7),nn.ReLU(),nn.Unflatten(1,(64,7,7)),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(32, 1, kernel_size=3, padding=1),nn.Sigmoid()
                                    )

    def reparam(selfs,u,lv):
        std = torch.exp(0.5*lv)
        eps = torch.randn_like(std)
        z = u+eps*std
        return z

    def encoder(self,x):
        x = self.encode(x)
        self.lv = self.lvn(x)
        self.u = self.un(x)
        z = self.reparam(self.u,self.lv)
        return z ,self.u,self.lv
    def decoder(self,z):
        return self.decode(z)


    def forward(self,x):
        z,u,lv = self.encoder(x)
        recon = self.decoder(z)
        return recon , u,lv

class Loss_f(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = args.a_kl
    def forward(self,x,recon,u,lv):
        l_re = F.binary_cross_entropy(recon,x,reduction='sum')
        lv = torch.clamp(lv, min=-10.0, max=10.0)  # bound logvar to curb exploding gradients
        l_kl = -0.5 * torch.sum(torch.sum(1 + lv - u.pow(2) - lv.exp(), dim=1))
        l_tatal= (l_re+self.a*l_kl)/x.size(0)
        return l_tatal,l_re/x.size(0),l_kl/x.size(0)
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

def sample(model_path='VAE_best.pth', num_samples=args.num_sam, device=args.device, out_path='samples.png'):
    """Load a trained VAE and generate samples to an image file."""
    device = device
    net = VAE().to(device)
    state = torch.load(model_path, map_location=device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, nn.Module):
            net = checkpoint
        else:
            net.load_state_dict(checkpoint)
    net.eval()
    z = torch.randn(num_samples, 128, device=device)
    imgs = net.decoder(z)
    save_image(imgs, out_path, nrow=max(1, int(num_samples ** 0.5)), normalize=True)
    return out_path


def sample_variations(model_path='VAE_best.pth', num_variations=5, base_image_idx=0,
                      device=None, out_path='variations.png'):
    """生成单个图像的变化版本（在潜在空间中插值）"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    net = VAE().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, nn.Module):
            net = checkpoint
        else:
            net.load_state_dict(checkpoint)
    net.eval()
    # 加载测试集中的一个图像
    _, test_loader = dataloader()
    test_data = next(iter(test_loader))
    base_img = test_data[0][base_image_idx:base_image_idx + 3].to(device)

    with torch.no_grad():
        # 编码得到潜在向量
        z_base, _, _ = net.encoder(base_img)

        # 生成变化
        variations = []
        for i in range(num_variations):
            # 添加随机噪声
            z_variation = z_base + torch.randn_like(z_base) * 0.5
            img_variation = net.decoder(z_variation)
            variations.append(img_variation)

        # 组合图像（原图+变化）
        all_imgs = torch.cat([base_img] + variations, dim=0)
        save_image(all_imgs, out_path, nrow=num_variations + 1, normalize=True)

    print(f"已生成 {num_variations} 个变体到: {out_path}")
    return out_path
def train():
    net = VAE().to(args.device)
    loss_f = Loss_f().to(args.device)
    # 优化器
    if args.optim == 'Adam':
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        opt = torch.optim.SGD(net.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    train_loader,test_loader = dataloader()
    best_loss = float('inf')
    for epoch in range(args.epoches):
        train_loss = 0
        train_re = 0
        train_kl = 0
        test_loss = 0
        test_re = 0
        test_kl = 0
        pbar1 = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoches} [Train]')
        pbar2 = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{args.epoches} [Test]')
        net.train()
        for features,_ in pbar1:
            features = features.to(args.device)
            opt.zero_grad()
            recon,u,lv = net(features)
            l,l_re,l_kl = loss_f(features,recon,u,lv)
            l.backward()
            opt.step()
            train_loss +=l.item()
            train_re +=l_re.item()
            train_kl +=l_kl.item()
            pbar1.set_postfix({'Loss': l.item(), 'Recon': l_re.item(), 'KL': l_kl.item()})
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_re = train_re / len(train_loader.dataset)
        avg_train_kl = train_kl / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_re:.4f}, KL: {avg_train_kl:.4f}')
        writer.add_scalar('train_loss',avg_train_loss,epoch+1)
        writer.add_scalar('train_re',avg_train_re,epoch+1)
        writer.add_scalar('train_re',avg_train_kl,epoch+1)
        net.eval()
        for features, _ in pbar2:
            features = features.to(args.device)
            recon, u, lv = net(features)
            l, l_re, l_kl = loss_f(features, recon, u, lv)
            test_loss += l.item()
            test_re += l_re.item()
            test_kl += l_kl.item()

        avg_test_loss = test_loss / len(test_loader.dataset)
        avg_test_re = test_re / len(test_loader.dataset)
        avg_test_kl = test_kl / len(test_loader.dataset)

        print(f'Epoch {epoch + 1}: Test Loss: {avg_test_loss:.4f}, Recon: {avg_test_re:.4f}, KL: {avg_test_kl:.4f}')

        # 记录到TensorBoard
        writer.add_scalar('Loss/test', avg_test_loss, epoch + 1)
        writer.add_scalar('Recon/test', avg_test_re, epoch + 1)
        writer.add_scalar('KL/test', avg_test_kl, epoch + 1)

        if avg_test_loss<best_loss:
            best_ac= test_loss
            delete_checkpoint("VAE_best.pth")
            torch.save(net,"VAE_best.pth")


if __name__ == '__main__':
    train()
    sample()
    sample_variations(base_image_idx=55)













