from main import *
import networkx as nx
import torch

device = torch.device('cuda')

String_map = np.load('Data/Yeast/PIPR-cut/graph.emb_2039.npz')
embed_data = "Data/Yeast/PIPR-cut/Ks-coding/"

nw = 'Data/Yeast/PIPR-cut/nw2039.txt'
with open(nw,'r') as ff:
    name_pairs_lines = ff.readlines()
names = {}
for i in name_pairs_lines:
    names[i.strip().split('\t')[0][:-6]]=i.strip().split('\t')[1][:-6]


'''加载'''
def default_loader(pid):
    name1 = pid.split('_')[0]
    name2 = pid.split('_')[1]

    graph_data1 = String_map[names[name1]]
    graph_data_1 = torch.tensor(graph_data1).float().to(device)
    graph_data2 = String_map[names[name2]]
    graph_data_2 = torch.tensor(graph_data2).float().to(device)

    textembed1 = np.load(embed_data+name1+'.npy')
    textembed_1 = torch.tensor(textembed1).float().to(device)
    textembed2 = np.load(embed_data+name2+'.npy')
    textembed_2 = torch.tensor(textembed2).float().to(device)
    return graph_data_1,graph_data_2, textembed_1, textembed_2  # 输出图和补全的预训练特征


class MyDataset(Dataset):

    def __init__(self, pairs, loader=default_loader):
        super(MyDataset, self).__init__()
        self.pns = pairs
        self.loader = loader

    def __getitem__(self, index):
        p1, label = self.pns[index]
        G1,G2, embed1, embed2 = self.loader(p1)  # default_loader：alphafold_cmap/中的npz
        return G1,G2, embed1, embed2, label

    def __len__(self):
        return len(self.pns)
