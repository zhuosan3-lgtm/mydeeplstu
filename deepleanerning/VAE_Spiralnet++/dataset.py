import os.path

import torch
from torch import nn
import trimesh
import glob
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class dataset():
    def __init__(self,data_root='MPI-FAUST/training/registrations',split='train',seq_length=9):
        self.data_path = data_root
        self.split = split
        self.seq_length = seq_length

        self.sample = self.load_sample()
        self.mesh = trimesh.load(self.sample[0])

        self.adj = self.get_adj()
        self.sequences = self.get_all_sequence()
        self.all_mesh = self.get_all_mesh()
        # print(self.sequences.shape)
        # print(self.all_mesh.shape)

    def load_sample(self):
        ply_file = glob.glob(os.path.join(self.data_path,'*.ply'),recursive=True)
        # for file in ply_file:
        #     print(os.path.basename(file))
        total_len = len(ply_file)
        end_train = int(0.7*total_len)
        end_val = int(0.85*total_len)
        if self.split =='train':
            return ply_file[:end_train]
        elif self.split =='val':
            return ply_file[end_train:end_val]
        else:
            return ply_file[end_val:]
    def get_adj(self):
        adj = [[] for _ in range(len(self.mesh.vertices))]
        for i,face in enumerate(self.mesh.faces):
            v0,v1,v2 = face
            if v0 not in adj[v1]:
                adj[v1].append(v0)
            if v0 not in adj[v2]:
                adj[v2].append(v0)
            if v1 not in adj[v0]:
                adj[v0].append(v1)
            if v1 not in adj[v2]:
                adj[v2].append(v1)
            if v2 not in adj[v1]:
                adj[v1].append(v2)
            if v2 not in adj[v0]:
                adj[v0].append(v2)
        return adj
    def get_one_sequence(self,id):
        sequence = [id]
        #visit = set([id])
        current = [id]
        while len(sequence) < self.seq_length:
            next_ring = []
            for ver in current:
                for neighbor in self.adj[ver]:
                    if neighbor not in sequence:
                        sequence.append(neighbor)
                        #visit.add(neighbor)
                        next_ring.append(neighbor)
                        if len(sequence) >= self.seq_length:
                            break
                if len(sequence) >= self.seq_length:
                    break
            if not next_ring:
                break
            current = next_ring
        while len(sequence) < self.seq_length:
            sequence.append(sequence[-1])
        return sequence
    def get_all_sequence(self):
        sequences = []
        for i in range(len(self.adj)):
            sequence = self.get_one_sequence(i)
            sequences.append(sequence)
        sequences = torch.tensor(sequences)
        return sequences
    def get_all_mesh(self):
        meshes = []
        for i in self.sample:
            mesh = trimesh.load(i)
            meshes.append(mesh.vertices)
        meshes = torch.tensor(meshes)
        return meshes
    def __len__(self):
        return len(self.sample)
    def __getitem__(self, idx):
        """加载单个样本"""
        mesh_path = self.sample[idx]

        # 加载网格
        mesh = trimesh.load(mesh_path)
        vertices = torch.FloatTensor(mesh.vertices)  # [6890, 3]


        # 生成样本ID（从文件名提取）
        filename = os.path.basename(mesh_path)
        sample_id = filename.replace('tr_reg_', '').replace('.ply', '')

       # sample = {
            #'vertices': vertices,  # 顶点坐标 [6890, 3]
            #'faces': torch.LongTensor(mesh.faces),  # 面片信息
            #'spiral_indices': self.sequences,  # 螺旋索引 [6890, seq_len]
            #'sample_id': sample_id,  # 样本ID
            #'filename': filename  # 文件名
        #}
        sample = vertices

        return sample
if __name__== '__main__':
    data = dataset()






