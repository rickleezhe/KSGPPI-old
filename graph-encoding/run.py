import networkx as nx
import  node2vec

with open('Data/Yeast/PIPR-cut/nw2039.txt','r') as ff:
    name_pairs_lines = ff.readlines()
names = {}
for i in name_pairs_lines:
    names[i.strip().split('\t')[0][:-6]]=i.strip().split('\t')[1][:-6]
c_s = 0
interaction_file = 'Data/Yeast/4932.protein.physical.links.v11.5.txt'
G = nx.DiGraph()  #有向图
with open(interaction_file, 'r') as file:
    for line_number, line in enumerate(file):
        if line_number == 0:
            continue  # 跳过第一行
        parts = line.strip().split()  # 分割每行数据
        # print(parts)
        # exit()
        if len(parts) == 3:
            node1, node2, edge_feature = parts
            if float(edge_feature) >= c_s:
                G.add_node(node1)
                G.add_node(node2)
                G.add_edge(node1, node2, weight=1.0)  # 假设边特征是浮点数

edge_count = G.number_of_edges()
print(f"图的边数: {edge_count}")



interaction_file1 = 'Data/Yeast/PIPR-cut/PIPR_cut_2039.txt'
G1 = nx.DiGraph()  #有向图
with open(interaction_file1, 'r') as file1:
    for line_number, line in enumerate(file1):
        parts = line.strip().split()  # 分割每行数据
        if len(parts) == 3:
            node1, node2, edge_feature = parts
            G1.add_node(names[node1])
            G1.add_node(names[node2])
            G1.add_edge(names[node1], names[node2], weight=1.0)  # 假设边特征是浮点数
            G1.add_edge(names[node2], names[node1], weight=1.0)  # 假设边特征是浮点数

edge_count1 = G1.number_of_edges()
print(f"图的边数: {edge_count1}")

G.remove_edges_from(G1.edges())

edge_count = G.number_of_edges()
print(f"图的边数: {edge_count}")
#exit()

model = node2vec.Node2vec(G,path_length=64,num_paths=32,p=1,q=1)

model.train(dim=100,workers=8,window_size=10)
model.save_embeddings('Data/Yeast/PIPR-cut/graph.emb_2039.npz')

