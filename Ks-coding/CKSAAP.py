import torch
import torch.nn as nn
import math
from torch.autograd import Variable

from itertools import product

class CKSAAP(nn.Module):
    def __init__(self, is_use_position=False, position_d_model=None):
        super(CKSAAP, self).__init__()

        AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        DP = list(product(AA, AA))
        # print(DP) #[('A', 'A'), ('A', 'C'), ..., ('Y', 'W'), ('Y', 'Y')]
        # print(len(DP)) #400
        self.DP_list = []
        for i in DP:
            self.DP_list.append(str(i[0]) + str(i[1]))
        # print(self.DP_list)  #['AA', 'AC', ..., 'YW', 'YY']

        self.position_func = None
        self.position_d_model = position_d_model
        if is_use_position:
            if None is position_d_model:
                self.position_d_model = 16
            self.position_func = PositionalEncoding(d_model=self.position_d_model)

    def returnCKSAAPcode(self, query_seq, k=3):
        code_final = []
        for turns in range(k + 1):
            DP_dic = {}
            code = []
            code_order = []
            for i in self.DP_list:
                DP_dic[i] = 0
            for i in range(len(query_seq) - turns - 1):
                tmp_dp_1 = query_seq[i]
                tmp_dp_2 = query_seq[i + turns + 1]
                tmp_dp = tmp_dp_1 + tmp_dp_2
                if tmp_dp in DP_dic.keys():
                    DP_dic[tmp_dp] += 1
                else:
                    DP_dic[tmp_dp] = 1
            for i, j in DP_dic.items():
                code.append(j / (len(query_seq) - turns - 1))
            for i in self.DP_list:
                code_order.append(code[self.DP_list.index(i)])
            code_final += code

        code_final = torch.FloatTensor(code_final)
        code_final = code_final.view(k+1, 20, 20)
        return code_final

    def return_CKSAAP_Emb_code(self, query_seq, emb, k=3, is_shape_for_3d=False):
        """
        :param is_shape_for_3d:
        :param query_seq: L
        :param emb: [L, D] tensor
        :param k:
        :return:
        """
        code_final = []
        for turns in range(k + 1):
            DP_dic = {}
            code = []
            code_order = []
            for i in self.DP_list: ##['AA', 'AC', ..., 'YW', 'YY']
                DP_dic[i] = torch.zeros(emb.size(-1))
            # print(DP_dic) #{'AA': tensor([0., 0., 0.,  ..., 0., 0., 0.]),..., 'YY': tensor([0., 0., 0.,  ..., 0., 0., 0.])}  #1280
            for i in range(len(query_seq) - turns - 1):
                tmp_dp_1 = query_seq[i]
                tmp_dp_2 = query_seq[i + turns + 1]
                tmp_emb_1 = emb[i]
                tmp_emb_2 = emb[i + turns + 1]
                tmp_emb = 0.5 * (tmp_emb_1 + tmp_emb_2)

                tmp_dp = tmp_dp_1 + tmp_dp_2
                if tmp_dp in DP_dic.keys():
                    DP_dic[tmp_dp] += tmp_emb
                else:
                    DP_dic[tmp_dp] = tmp_emb
            for i, j in DP_dic.items():
                code.append(j / (len(query_seq) - turns - 1))
            for i in self.DP_list:
                code_order.append(code[self.DP_list.index(i)])
            print(len(code))
            print(code[0].shape)
            code_final += code

        code_final = torch.stack(code_final)
        code_final = code_final.view(k+1, 20, 20, -1)

        if is_shape_for_3d:
            k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = code_final.size()
            code_final = code_final.permute(0, 3, 1, 2).contiguous().\
                view(k_plus_one * position_posi_emb_size, aa_num_1, aa_num_2)
        return code_final

    def return_CKSAAP_position_code(self, query_seq, k=3):
        """
        :param query_seq: L
        :param embs: [L, D] tensor
        :param k:
        :return: [(k+1)*position_posi_emb_size, 20, 20]
        """
        posi_emb = self.position_func(
            torch.zeros(1, len(query_seq), self.position_d_model)
        ).squeeze(0)

        # [(k+1), 20, 20, position_posi_emb_size]
        emb = self.return_CKSAAP_Emb_code(query_seq, posi_emb, k)

        # [(k+1), 20, 20, position_posi_emb_size] --> [(k+1)*position_posi_emb_size, 20, 20]
        k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = emb.size()
        emb = emb.permute(0, 3, 1, 2).contiguous().view(k_plus_one*position_posi_emb_size, aa_num_1, aa_num_2)
        return emb

with open('Data/Yeast/PIPR-cut/PIPR_cut_2039_seq.tsv','r') as ff:
    fasta_lines = ff.readlines()
fasta = {}
for j in fasta_lines:
    fasta[j.strip().split('\t')[0]] = j.strip().split('\t')[1]

import numpy as np
embed_data = np.load("Data/Yeast/PIPR-cut/PIPR_cut_2039—esm2.npz")

model = CKSAAP()

#K20_representations = {}

import numpy as np

n = 0
for name in fasta:
    seq1 = fasta[name]
    textembed1 = torch.tensor(embed_data[name])
    textembed_1 = model.return_CKSAAP_Emb_code(seq1, textembed1, is_shape_for_3d=True)

    np.save('Data/Yeast/PIPR-cut/Ks-coding/'+name+'.npy', textembed_1)
    print(textembed_1.shape)
    n+=1
    print(n,name)


