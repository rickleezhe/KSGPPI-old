import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader, Dataset
import sys
from graph_data_loader import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from TAGlayer import *

import time
import random
import numpy as np
import torch
import os
seed = 123  # 设置种子值，可以是任意整数
# 设置 Python 内置随机数生成器的种子
random.seed(seed)
# 设置 NumPy 随机数生成器的种子
np.random.seed(seed)
# 设置 PyTorch 随机数生成器的种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 如果使用了 CuDNN 加速，则还需要设置这两个选项
torch.backends.cudnn.deterministic = True #每次返回的卷积算法将是确定
torch.backends.cudnn.benchmark = False


def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (G1,G2,dmap1, dmap2, y) in enumerate(loader):
            output = model(G1,G2,dmap1, dmap2)
            output = torch.round(output.squeeze(1))
            total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def train(trainArgs,nn):
    train_losses = []
    train_accs = []
    bestmcc=0.
    for i in range(trainArgs['epochs']):  # 50
        print(nn,"Running EPOCH", i + 1)
        since = time.time()
        total_loss = 0
        n_batches = 0
        correct = 0
        train_loader = trainArgs['train_loader']  # train_loader--在graph_cmap_loader中
        optimizer = trainArgs['optimizer']  # Adam
        criterion = trainArgs["criterion"]  # BCEloss
        attention_model = trainArgs['model']


        for batch_idx, (G1, G2,dmap1, dmap2, y) in enumerate(train_loader):
            y_pred = attention_model(G1,G2,dmap1, dmap2)

            correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                y.type(torch.DoubleTensor)).data.sum()
            loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))
            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_batches += 1

        avg_loss = total_loss / n_batches
        acc = correct.numpy() / (len(train_loader.dataset))

        train_losses.append(avg_loss)
        train_accs.append(acc)

        print(nn,"train avg_loss is", avg_loss)
        print(nn,"train ACC = ", acc)

        # if (trainArgs['doSave']):
        #     torch.save(attention_model.state_dict(), pkl_path + 'epoch' + '%d.pkl' % (i + 1))
        # test
        total_labels, total_preds = predicting(attention_model, test_loader)
        print(total_labels.shape)
        from util import calculateMaxMCCThresOnValidset
        metric, roc_data, prc_data, corrThres = calculateMaxMCCThresOnValidset(total_preds, total_labels, is_valid=False,test_thre=0.5, draw=False)
        print('ACC', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC', 'AUCPR', 'maxMCC', 'corrTP', 'corrFP',
              'corrTN', 'corrFN')
        print(['{:.10f}'.format(num) for num in metric])
        if metric[7] >= bestmcc:
            bestmcc = metric[7]
            torch.save(attention_model.state_dict(), os.path.dirname(rst_file) + '/epoch_'+str(nn)+ '.pkl')
            metric_mcc = metric
        with open(rst_file, 'a+') as fp:
            fp.write(str(nn)+'_epoch:' + str(i+1) + '\ttrainacc=' + str(acc) +'\ttrainloss=' + str(avg_loss.item()) +'\tacc=' + str(metric[0].item()) + '\tprec=' + str(metric[1].item()) + '\trecall=' + str(metric[2].item()) +  '\tf1=' + str(metric[4].item()) + '\tauc=' + str(metric[5].item()) + '\tspec='+str(metric[3].item())+ '\tmcc='+str(metric[7].item())+'\n')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return metric_mcc

if __name__ == "__main__":
    epoch = 30
    batchsize = 32

    datasetname = 'yeast'
    rst_file = './results/yeast_results.tsv'
    batchsize = int(batchsize)

    modelArgs = {}
    modelArgs['emb_dim'] = 5120
    modelArgs['dropout'] = 0.2

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    def loadStrMtx(mtx_path):
        import re
        with open(mtx_path, 'r') as fh:
            pas = []
            lbs = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split('  |\t', line)
                pas.append(words[0] + '_' + words[1])  # [('P39108', 'P14908', 0)]
                lbs.append(int(words[2]))
        return pas, lbs

    pair_path = 'Data/Yeast/PIPR-cut/PIPR_cut_2039.txt' #配对及标签
    X, y = loadStrMtx(pair_path)

    mcc_mean = 0.
    acc_mean = 0.
    auc_mean = 0.
    prec_mean = 0.
    spec_mean = 0.
    recall_mean = 0.
    f1_mean = 0.
    end_fold = 5
    current_fold = 0

    fold_counter = 0  # 用于跟踪当前折的计数器

    for train_index, test_index in skf.split(X, y):
        print(f"Fold {fold_counter + 1}")
        print("Train indices:", train_index)
        print("Test indices:", test_index)

        if fold_counter >= current_fold and end_fold > fold_counter:
            trainArgs = {}
            trainArgs['epochs'] = int(epoch)
            trainArgs['lr'] = 0.001
            trainArgs['model'] = TemGPPI(modelArgs).cuda()
            print(trainArgs['model'])
            
            trainArgs['optimizer'] = torch.optim.AdamW(trainArgs['model'].parameters(), lr=trainArgs['lr'])
            trainArgs['criterion'] = torch.nn.BCELoss()
            # trainArgs['lr_scheduler'] = torch.optim.lr_scheduler.StepLR(trainArgs['optimizer'], step_size=20, gamma=0.1,last_epoch=-1)
            nn = fold_counter+1
            tr_pairs = [[X[i], y[i]] for i in train_index]
            tst_pairs = [[X[i], y[i]] for i in test_index]
            np.save(os.path.dirname(rst_file) + '/'+str(nn)+'.npy', tst_pairs)  # 保存为.npy格式
            train_dataset = MyDataset(tr_pairs)
            trainArgs['train_loader'] = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)

            test_dataset = MyDataset(tst_pairs)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=True, drop_last=False)

            metric_mcc_one = train(trainArgs,nn)

            mcc_mean += metric_mcc_one[7].item()
            acc_mean += metric_mcc_one[0].item()
            prec_mean += metric_mcc_one[1].item()
            spec_mean += metric_mcc_one[3].item()
            recall_mean += metric_mcc_one[2].item()
            f1_mean += metric_mcc_one[4].item()
            auc_mean += metric_mcc_one[5].item()
        fold_counter += 1

    print('mcc_mean', mcc_mean/5,'acc_mean', acc_mean/5, 'prec_mean', prec_mean/5, 'recall_mean', recall_mean/5, 'spec_mean', spec_mean/5, 'f1_mean', f1_mean/5,'auc_mean', auc_mean/5)
    with open(rst_file, 'a+') as f:
        f.write(f'\n\nmcc_mean:{"%.6f" % (mcc_mean/5)},acc_mean: {"%.6f"%(acc_mean/5)}, prec_mean: {"%.6f"%(prec_mean/5)}, recall_mean:{"%.6f" % (recall_mean/5)}, spec_mean: {"%.6f" % (spec_mean/5)}, f1_mean: {"%.6f" % (f1_mean/5)},auc_mean: {"%.6f"%(auc_mean/5)}\n')
