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
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--lr', type=float, default=0.001)  # 改小学习率
    parse.add_argument('--epoches', type=int, default=10)
    parse.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parse.add_argument('--optim', default='Adam')
    return parse.parse_args()


args = Parse()
writer = SummaryWriter('loss5')


def dataloader():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data', train=True, transform=transform, download=True)
    test_set = datasets.MNIST('./data', train=False, transform=transform, download=True)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU()
        )

        # 均值和对数方差
        self.fc_mu = nn.Linear(1024, 128)
        self.fc_logvar = nn.Linear(1024, 128)

        self.decode = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x):
        encoded = self.encode(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decoder(self, z):
        return self.decode(z)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar


class Loss_f(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1.0  # KL散度权重，通常设为1.0

    def forward(self, x, recon, mu, logvar):
        # 重建损失
        l_re = F.binary_cross_entropy(recon, x, reduction='sum')

        # KL散度损失
        l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 总损失
        total_loss = (l_re + self.a * l_kl) / x.size(0)  # 除以batch_size得到平均损失

        return total_loss, l_re / x.size(0), l_kl / x.size(0)


def delete_checkpoint(checkpoint_path):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"已删除旧检查点文件: {checkpoint_path}")


def sample(model_path='VAE_best.pth', num_samples=16, device=None, out_path='samples.png'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VAE().to(device)

    # 加载模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, nn.Module):
            net = checkpoint
        else:
            net.load_state_dict(checkpoint)
    else:
        print(f"模型文件 {model_path} 不存在")
        return None

    net.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 128, device=device)
        imgs = net.decoder(z)
        save_image(imgs, out_path, nrow=int(num_samples ** 0.5), normalize=True)
    print(f"样本已保存到: {out_path}")
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

    train_loader, test_loader = dataloader()
    best_loss = float('inf')

    for epoch in range(args.epoches):
        # 训练阶段
        net.train()
        train_loss = 0.0
        train_re = 0.0
        train_kl = 0.0

        pbar1 = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoches} [Train]')
        for features, _ in pbar1:
            features = features.to(args.device)
            opt.zero_grad()
            recon, mu, logvar = net(features)
            l, l_re, l_kl = loss_f(features, recon, mu, logvar)
            l.backward()
            opt.step()

            train_loss += l.item() * features.size(0)
            train_re += l_re.item() * features.size(0)
            train_kl += l_kl.item() * features.size(0)

            pbar1.set_postfix({'Loss': l.item(), 'Recon': l_re.item(), 'KL': l_kl.item()})

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_re = train_re / len(train_loader.dataset)
        avg_train_kl = train_kl / len(train_loader.dataset)

        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_re:.4f}, KL: {avg_train_kl:.4f}')

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
        writer.add_scalar('Recon/train', avg_train_re, epoch + 1)
        writer.add_scalar('KL/train', avg_train_kl, epoch + 1)

        # 测试阶段
        net.eval()
        test_loss = 0.0
        test_re = 0.0
        test_kl = 0.0

        with torch.no_grad():
            pbar2 = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{args.epoches} [Test]')
            for features, _ in pbar2:
                features = features.to(args.device)
                recon, mu, logvar = net(features)
                l, l_re, l_kl = loss_f(features, recon, mu, logvar)

                test_loss += l.item() * features.size(0)
                test_re += l_re.item() * features.size(0)
                test_kl += l_kl.item() * features.size(0)

        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_loader.dataset)
        avg_test_re = test_re / len(test_loader.dataset)
        avg_test_kl = test_kl / len(test_loader.dataset)

        print(f'Epoch {epoch + 1}: Test Loss: {avg_test_loss:.4f}, Recon: {avg_test_re:.4f}, KL: {avg_test_kl:.4f}')

        # 记录到TensorBoard
        writer.add_scalar('Loss/test', avg_test_loss, epoch + 1)
        writer.add_scalar('Recon/test', avg_test_re, epoch + 1)
        writer.add_scalar('KL/test', avg_test_kl, epoch + 1)

        # 保存最佳模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            delete_checkpoint("VAE_best.pth")

            # 保存完整模型
            torch.save(net, "VAE_best.pth")
            # 或者保存state_dict（推荐）
            # torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'best_loss': best_loss}, "VAE_best.pth")

            print(f"保存最佳模型，测试损失: {best_loss:.4f}")

    writer.close()
    print("训练完成！")


if __name__ == '__main__':
    #train()
    sample()