# -*- coding: UTF-8 -*-
import esm
import torch.nn as nn
import torch
import numpy as np

class ESMFineTune(nn.Module):
    def __init__(self, esm2_layer=0):
    # def __init__(self, esm2_layer=0,mid_dim=320):
        super(ESMFineTune, self).__init__()

        self.esm2_layer = esm2_layer

        self.Esm2, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        self.tokenizer1 = self.alphabet.get_batch_converter()
        self.esm2_freeze()


    def esm2_freeze(self):
        # 创建所有网络层字典{父层1:[子层1, 子层2, ...], 父层2:[子层1, 子层2, ...], ...}
        esm2_all_layer_dict = {}
        # 变量num
        esm2_num = 0
        # 参数未冻结的网络层字典{父层1:[子层1, 子层2, ...], 父层2:[子层1, 子层2, ...], ...}
        esm2_unfreeze_layer_dict = {}

        for name, param in self.Esm2.named_parameters():
            param.requires_grad = False
        # 遍历esm2网络框架
        for name, param in self.Esm2.named_parameters():
            # 在all_layer_dict添加layers.n父层，字典的键为父层名称，值为子层列表
            if "layers" in name:
                key = name[0: name.find('.', 7)]
                if key in esm2_all_layer_dict.keys():
                    if key in name:
                        esm2_all_layer_dict[key].append(name)
                else:
                    esm2_all_layer_dict[key] = []
                    if key in name:
                        esm2_all_layer_dict[key].append(name)
        # 再次遍历esm2网络框架得到不需要冻结参数的网络层添加到字典unfreeze_layer_dict中
        for i, name in enumerate(esm2_all_layer_dict):
            if len(esm2_all_layer_dict) - self.esm2_layer < 0:
                exit("层数溢出")
            if esm2_num >= len(esm2_all_layer_dict) - self.esm2_layer:
                esm2_unfreeze_layer_dict[name] = esm2_all_layer_dict[name]
            esm2_num += 1

        # 未冻结参数个数
        not_freeze_param_nums_esm2 = 0
        # 遍历网络框架，冻结部分父层，并计算未冻结的父层参数总个数
        for name, param in self.Esm2.named_parameters():
            if name[0: name.find('.', 7)] in esm2_unfreeze_layer_dict.keys():
                param.requires_grad = True
                not_freeze_param_nums_esm2 += param.numel()
        print("esm2未冻结的参数个数为：", not_freeze_param_nums_esm2)

    def forward(self, x):
        _, _, tokens = self.tokenizer1(x)
        with torch.no_grad():
            representations = self.Esm2(tokens, repr_layers=[33], return_contacts=True)["representations"][33]
        mean_representations = representations[:, 1: len(x[0][1]) + 1].squeeze().numpy()
        return mean_representations


def loadStrMtx(mtx_path):
    with open(mtx_path, 'r') as f:
        lines = f.readlines()
    contents = [x.split() for x in lines]
    return contents



seq_path = 'Data/Yeast/PIPR-cut/PIPR_cut_2039_seq.tsv'  #序列
Model = ESMFineTune(esm2_layer=0)  # .to(device)
seq_dict = {}
esm_representations = {}
seq_mtx = loadStrMtx(seq_path)
# print(seq_mtx)  #id,序列
print(len(seq_mtx)) #2497条
n = 0
for vec in seq_mtx:
    n += 1
    if 100<n<120:
        print(vec[0],len(vec[1]))
    seq_dict[vec[0]] = vec[1]  #字典id:seq
    seq_esm_input = [(vec[0], vec[1])]  #[:1200]string
    out = Model(seq_esm_input)
    esm_representations[vec[0]] = out
    print(n)
    # print(seq_dict)
    # print(esm_representations)
    # print(esm_representations['P52918'].shape)
    # exit()

np.savez('Data/Yeast/PIPR-cut/PIPR_cut_2039—esm2.npz', **esm_representations)

