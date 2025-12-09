from torch import nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim * 2)  # 输出均值和对数方差
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()  # 输出在0-1之间
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(z)
        return x_reconst, mu, log_var



