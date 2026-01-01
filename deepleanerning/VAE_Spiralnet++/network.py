from torch import nn
import torch
import torch.nn.functional as F

class SpiralConv(nn.Module):
    def __init__(self,in_channels,out_channels,indices,dim=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.dim = dim
        self.indices = indices
        self.layer = nn.Linear(self.in_channels*self.seq_length,self.out_channels)
        self.re_parameter()

    def re_parameter(self):
        nn.init.xavier_uniform(self.layer.weight)
        #nn.init.constant(self.layer,0)
        nn.init.zeros_(self.layer.bias)
    def forward(self,x):
        num = self.indices.size(0)
        if self.dim == 3 :
            bs = x.size(0)
            flat_indices= self.indices.view(-1)
            x = torch.index_select(x,1,flat_indices)
            x = x.view(bs,num,-1)
        elif self.dim == 2:
            flat_indices = self.indices.view(-1)
            x = torch.index_select(x,1,flat_indices)
            x = x.view(num,-1)
        else:
            print('error dim=2 or 3')
        x = self.layer(x)
        return x
class VAE(nn.Module):
    def __init__(self,in_channels,indices,latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.indices = indices
        self.num = indices.size(0)
        self.seq_length = indices.size(1)
        self.latent_dim = latent_dim
        self.encode = nn.Sequential(
            SpiralConv(self.in_channels,64,self.indices),nn.ReLU(),
            SpiralConv(64,128,self.indices),nn.ReLU(),
            SpiralConv(128, 256, self.indices), nn.ReLU(),
            SpiralConv(256, 512, self.indices), nn.ReLU(),
            #nn.Flatten(),nn.Linear(128*self.num,1024),nn.ReLU()
        )
        self.mu_n = nn.Linear(512,self.latent_dim)
        self.lv_n = nn.Linear(512,self.latent_dim)
        self.d1 = nn.Linear(self.latent_dim,64)
        self.decode = nn.Sequential(
            SpiralConv(128, 256, self.indices), nn.ReLU(),
            SpiralConv(256, 128, self.indices), nn.ReLU(),
            SpiralConv(128,64,self.indices),nn.ReLU(),
            SpiralConv(64,self.in_channels,self.indices)
        )
        self.decode1 = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num * 64),
            #nn.Linear(128, self.num * 16),
            # nn.ReLU(inplace=True),
            # nn.Linear(64, self.num * 3)  # 输出所有顶点坐标
        )
        self.decode2 = nn.Sequential(
            # SpiralConv(16, 32,self.indices),
            # nn.ReLU(inplace=True),
            # SpiralConv(32, 64,self.indices),
            # nn.ReLU(inplace=True),
            SpiralConv(64, 32,self.indices),
            nn.ReLU(inplace=True),
            SpiralConv(32, 16, self.indices),
            nn.ReLU(inplace=True),
            SpiralConv(16, 3,self.indices)  # 输出所有顶点坐标
        )
        self.decode3 = nn.Sequential(
            # SpiralConv(1024, 256, self.indices),
            # nn.ReLU(inplace=True),
            SpiralConv(128, 64, self.indices),
            nn.ReLU(inplace=True),
            SpiralConv(64, 16, self.indices),
            nn.ReLU(inplace=True),
            SpiralConv(16, 3, self.indices)  # 输出所有顶点坐标
        )

    def re_para(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu+eps*std
        return z
    def encoder(self,x):
        x = self.encode(x)
        #x = x.mean(dim=1)
        x = x.max(dim=1).values
        mu = self.mu_n(x)
        log_var = self.lv_n(x)
        z = self.re_para(mu,log_var)
        #print(z.size())
        return z,mu,log_var
    def decoder(self,z):
        x = self.d1(z)
        y = x.unsqueeze(1).repeat(1, self.num, 1)
        bs = x.size(0)
        # x = x.view(bs,self.num,128)
        #print(x.size())
        x = self.decode1(x)
        x = x.view(bs,self.num,-1)
        # x = self.decode2(x)
        cat = torch.cat((x,y),dim=2)
        cat = self.decode3(cat)

        return cat
    def forward(self,x):
        z,mu,log_var = self.encoder(x)
        recon = self.decoder(z)
        return recon,mu,log_var
class Loss_f(nn.Module):
    def __init__(self,kl_a=1):
        super().__init__()
        self.a = kl_a
    def forward(self,x,recon,mu,log_var):
        l_recon = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        #l_recon = nn.MSELoss(recon,x,'sum')/x.size(0)
        l_kl = -0.5 * torch.sum(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))/x.size(0)
        l_total = l_recon+self.a*l_kl
        return l_total,l_recon,l_kl
















