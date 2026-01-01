import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from dataset import dataset
import seaborn as sns
from network import VAE ,Loss_f
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from torch.utils import data
def Parse():
    parse = argparse.ArgumentParser("VAE模型参数设置")
    parse.add_argument('--batch_size',type=int,default=5)
    parse.add_argument('--lr',default=0.0001)
    parse.add_argument('--epoches',type=int,default=500)
    parse.add_argument('--device',type=str,default='cuda')
    parse.add_argument('--optim',default=torch.optim.Adam)
    parse.add_argument('--a_kl',default=1)
    parse.add_argument('--num_sam',type=int,default=32)
    parse.add_argument('--inter_step',type=int,default=5)
    return parse.parse_args()

args = Parse()
writer = SummaryWriter('loss2')
def dataloader(root='MPI-FAUST/training/registrations'):
    train_set = dataset(data_root=root,split='train',seq_length=9)
    val_set = dataset(data_root=root,split='val',seq_length=9)
    test_set = dataset(data_root=root,split='test',seq_length=9)
    indices = train_set.sequences
    faces = train_set.mesh.faces

    train_loader = data.DataLoader(train_set,batch_size=args.batch_size)
    val_loader = data.DataLoader(val_set,batch_size=args.batch_size)
    test_loader = data.DataLoader(test_set,batch_size=args.batch_size)
    return train_loader,val_loader,test_loader,indices,faces
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



def save_complete_mesh_simple(vertices, faces, filename):
    """简化版本 - 保存完整网络"""

    # 转换vertices
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()

    # 确保是二维数组
    if vertices.ndim == 3:
        vertices = vertices[0]  # 取batch维度
    elif vertices.ndim == 1:
        vertices = vertices.reshape(-1, 3)

    # 转换faces
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    # 写入OBJ文件
    with open(filename, 'w') as f:
        # 写入顶点
        for i in range(vertices.shape[0]):
            f.write(f"v {vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f}\n")

        # 写入面片
        for i in range(faces.shape[0]):
            f.write(f"f {int(faces[i, 0]) + 1} {int(faces[i, 1]) + 1} {int(faces[i, 2]) + 1}\n")

    print(f"完整网络已保存: {filename}")



def sample():
    train_loader, val_loader, test_loader, indices, faces = dataloader()

    net = torch.load("VAE_mesh_best.pth", weights_only=False).to('cuda')
    net.eval()

    with torch.no_grad():
        z = torch.randn((1,indices.size(0), 64), device='cuda')
        vertices = net.decoder(z)
    filename = 'sample.obj'
    save_complete_mesh_simple(vertices, faces, filename)
def visualize_vae_reconstruction_complete(device='cuda', epoch=500, num_samples=2,root="VAE_mesh_best.pth"):
    """可视化VAE重建效果（包含面片）"""
    train_loader, val_loader, test_loader, indices, faces = dataloader()
    Dataloader = val_loader
    model = torch.load(root, weights_only=False).to('cuda')
    model.eval()

    with torch.no_grad():
        vertices = next(iter(Dataloader)).to(device)

        #vertices = batch['vertices'][:num_samples].to(device)

        reconstructed, mu, logvar = model(vertices)

        # 计算重建误差
        # error = torch.sqrt(torch.sum((vertices - reconstructed) ** 2, dim=-1))
        # mean_error = error.mean(dim=1)
        #
        # print(f"  重建误差: {mean_error.cpu().numpy()}")

        # 保存原始和重建的完整网格
        for i in range(num_samples):
            original_verts = vertices[i].cpu().numpy()
            recon_verts = reconstructed[i].cpu().numpy()

            save_complete_mesh_simple(original_verts, faces, f'original_epoch_{epoch}_sample_{i}.obj')
            save_complete_mesh_simple(recon_verts, faces, f'reconstructed_epoch_{epoch}_sample_{i}.obj')
def save_inter(id1=0,id2=4):
    vertice1 = get_vertice(root='MPI-FAUST/training/registrations',split='train',index=id1)
    vertice2 = get_vertice(root='MPI-FAUST/training/registrations',split='train',index=id2)
    interpolate(vertice1, vertice2)



def interpolate(vertice1,vertice2,root="VAE_mesh_best2.pth"):
    train_loader, val_loader, test_loader, indices, faces = dataloader()
    model = torch.load(root, weights_only=False).to('cuda')
    model.eval()
    vertice1 = vertice1.to('cuda')
    vertice2 = vertice2.to('cuda')
    vertice1 = vertice1.reshape(1,indices.size(0), 3)
    vertice2 = vertice2.reshape(1,indices.size(0), 3)
    z1,mu1,log_var1 = model.encoder(vertice1)
    z2,mu1,log_var1 = model.encoder(vertice2)
    print(z1.size())
    alphas = torch.linspace(0,1,steps=args.inter_step).to('cuda')
    for alpha in alphas:
        z = (1-alpha.item())*z1+alpha.item()*z2
        vertices = model.decoder(z)
        filename = f'interpolate_{alpha}.obj'
        save_complete_mesh_simple(vertices, faces, filename)

def get_vertice(root='MPI-FAUST/training/registrations',split='train',index=0):
    if split=='train':
        train_set = dataset(data_root=root, split='train', seq_length=9)
        vertice = train_set[index]
        return vertice

    elif split == 'val':
        val_set = dataset(data_root=root, split='val', seq_length=9)
        vertice = val_set[index]
        return vertice

    else:
        test_set = dataset(data_root=root, split='test', seq_length=9)
        vertice = test_set[index]
        return vertice
def pac_var(root="VAE_mesh_best.pth"):
    train_loader, val_loader, test_loader, indices, faces = dataloader()
    model = torch.load(root, weights_only=False).to('cuda')
    model.eval()
    mus = []
    train_set = dataset(data_root='MPI-FAUST/training/registrations', split='train', seq_length=9)
    for i in train_set:
        i = i.to('cuda')
        i = i.reshape(1, indices.size(0), 3)
        z, mu, log_var = model.encoder(i)
        mus.append(mu.cpu())
    mus = torch.cat(mus,dim=0)


    # 假设你有数据 X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(mus.detach().numpy())

    # 执行PCA
    pca = PCA()  # 不指定n_components，保留所有主成分
    X_pca = pca.fit_transform(X_scaled)

    # 计算方差贡献率
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    print("各主成分方差贡献率:", explained_variance_ratio)
    print("累积方差贡献率:", cumulative_explained_variance_ratio)
    sns.set_style(rc= {'font.sans-serif':"Microsoft Yahei"})
    plt.figure(figsize=(10, 6))

    # 创建子图
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('主成分')
    plt.ylabel('方差贡献率')
    plt.title('各主成分方差贡献率')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1),
             cumulative_explained_variance_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%阈值')
    plt.axhline(y=0.85, color='g', linestyle='--', alpha=0.5, label='85%阈值')
    plt.xlabel('主成分数量')
    plt.ylabel('累积方差贡献率')
    plt.title('累积方差贡献率曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
def Pca(root="VAE_mesh_best.pth"):
    train_loader, val_loader, test_loader, indices, faces = dataloader()
    model = torch.load(root, weights_only=False).to('cuda')
    model.eval()
    mus = []
    train_set = dataset(data_root='MPI-FAUST/training/registrations', split='train', seq_length=9)
    for i in train_set:
        i = i.to('cuda')
        i = i.reshape(1, indices.size(0), 3)
        z, mu, log_var = model.encoder(i)
        mus.append(mu.cpu())
    mus = torch.cat(mus,dim=0)
    print(mus.size())
    pca = PCA(n_components=2)
    Z = pca.fit_transform(mus.detach().numpy())
    print(Z)
    plt.scatter(Z[:,0],Z[:,1])
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("Latent Space Pca")
    plt.show()
def train(root="VAE_mesh_best2.pth"):
    train_loader, val_loader, test_loader, indices,faces = dataloader()
    indices = indices.to(args.device)
    net = VAE(3,indices,32).to(args.device)
    loss_f = Loss_f(args.a_kl).to(args.device)
    if args.optim == 'Adam':
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        opt = torch.optim.SGD(net.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_loss = 1000000
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
        for features in pbar1:
            features = features.to(args.device)
            opt.zero_grad()
            recon, u, lv = net(features)
            l, l_re, l_kl = loss_f(features, recon, u, lv)
            l.backward()
            opt.step()
            train_loss += l.item()
            train_re += l_re.item()
            train_kl += l_kl.item()
            pbar1.set_postfix({'Loss': l.item(), 'Recon': l_re.item(), 'KL': l_kl.item()})
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_re = train_re / len(train_loader.dataset)
        avg_train_kl = train_kl / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_re:.4f}, KL: {avg_train_kl:.4f}')
        writer.add_scalar('train_loss', avg_train_loss, epoch + 1)
        writer.add_scalar('train_re', avg_train_re, epoch + 1)
        writer.add_scalar('train_re', avg_train_kl, epoch + 1)
        net.eval()
        for features in pbar2:
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

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            delete_checkpoint(root)
            torch.save(net, root)
            print('已更新最新模型')


if __name__ == '__main__':
    #train(root="VAE_mesh_best2.pth")
    #sample()
    #visualize_vae_reconstruction_complete(device='cuda', epoch=502, num_samples=5,root="VAE_mesh_best2.pth")
    #save_inter()
    #Pca("VAE_mesh_best2.pth")
    pac_var(root="VAE_mesh_best2.pth")


