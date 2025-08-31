import os
import scanpy as sc
import dgl
import random
import torch as th
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from SpaMIE.spamie_net import *
from SpaMIE.preprocess import preprocessing
from SpaMIE.create_graph import Sagegraph


def set_seed(rndseed, cuda: bool = True, extreme_mode: bool = True):
        os.environ["PYTHONHASHSEED"] = str(rndseed)
        random.seed(rndseed)
        np.random.seed(rndseed)
        th.manual_seed(rndseed)
        if cuda:
            th.cuda.manual_seed(rndseed)
            th.cuda.manual_seed_all(rndseed)
        if extreme_mode:
            th.backends.cudnn.benchmark = False
            th.backends.cudnn.deterministic = True
        dgl.seed(rndseed)
        dgl.random.seed(rndseed)

def sample_mask(idx, l):
    """Create mask."""
    mask = th.zeros(l)
    mask[idx] = 1
    return th.as_tensor(mask, dtype=th.bool)


def adjust_learning_rate(optimizer, epoch, start_lr1=0.02, start_lr2=0.02, lr_decay_rate=0.1, lr_decay_epochs=400, lr_decay_epochs2=400): # simu=0.5
    lr1 = start_lr1 * ( lr_decay_rate ** (epoch // lr_decay_epochs))
    param_group = optimizer.param_groups
    param_group[0]['lr'] =lr1
    param_group[1]['lr'] =lr1
    lr2 = start_lr2 * ( lr_decay_rate ** (epoch // lr_decay_epochs2))
    param_group[2]['lr'] =lr2



class Sagewrapper():

    def __init__(self, seed, device, in_feat, n_hidden, out_feat, datatype, type, sagetype, layers_nums, weight,
                  res_type=None, activation=None, dropout=0.2, epoch=350, lr=1e-4, lr2=0.01, batchnorm=True):
        super().__init__()

        self.seeds = seed
        self.device = device
        self.in_feat = in_feat
        self.n_hidden = n_hidden
        self.out_feat = out_feat
        self.activation = activation
        self.dropout = dropout
        self.type = type
        self.epoch = epoch
        self.datatype = datatype
        self.lr = lr
        self.lr2 = lr2
        self.layers_nums = layers_nums
        self.res_type = res_type
        self.wt = weight
        self.batchnorm = batchnorm
        self.sagetype = sagetype

        self.model = SpaMIE_net(self.in_feat, self.n_hidden, self.out_feat, self.wt,
                                           self.activation, self.sagetype, self.layers_nums, self.res_type, self.batchnorm, self.dropout).to(self.device)
        
        

    def data_split(self, feat_omics1, train_size):
        y = feat_omics1.shape[0]
        y = th.arange(y)
        sample_number = len(y)
        if train_size is not None:
            sample_idx = np.array(range(len(y)))
            train_idx = sample_idx[:train_size].tolist() 
            val_idx = sample_idx[int(0.7 * y.shape[0]): int(0.75 * y.shape[0])].tolist()
            test_idx = sample_idx[train_size:].tolist()

        else:
            sample_idx = shuffle(np.array(range(len(y))))
            val_idx = sample_idx[int(0.7 * y.shape[0]): int(0.75 * y.shape[0])].tolist()
            train_idx = sample_idx[:int(0.75 * y.shape[0])].tolist()
            test_idx = sample_idx[int(0.75 * y.shape[0]):].tolist()

        train_mask = sample_mask(train_idx, sample_number)
        val_mask = sample_mask(val_idx, sample_number)
        test_mask = sample_mask(test_idx, sample_number)

    
        return train_mask, val_mask, test_mask, train_idx, test_idx


    
    def fit(self, g_spatial_omics1, g_feature_omics1, adata_omics1, adata_omics2, output_dir, pred_name,
             true_name, train_size, weight=False, save_csv=False):
        
        set_seed(self.seeds)
        mse = nn.MSELoss()
        feat_omics1 = th.FloatTensor(adata_omics1.obsm['feat'].copy()).to(self.device)

        wt1_param_group = {'params': [self.model.wt1], 'lr': self.lr2}  
        wt2_param_group = {'params': [self.model.wt2], 'lr': self.lr2}  
        other_params = [param for param in self.model.parameters() if param is not self.model.wt1 and param is not self.model.wt2]

        optimizer = th.optim.AdamW([
            wt1_param_group,
            wt2_param_group,
            {'params': other_params, 'lr': self.lr}  
        ])


        train_mask,val_mask,test_mask,train_idx, test_idx = self.data_split(feat_omics1, train_size = train_size) 
        
        
        omics2_X = adata_omics2.X.copy()
        omics2_X = omics2_X
        omics2_X = th.from_numpy(omics2_X).to(self.device)
        omics2_X = omics2_X.float()
       
        vals = []
        for epoch in range(self.epoch):
            self.model.train()
            optimizer.zero_grad()
            if weight:
                output, wt, alph, latents = self.model(g_spatial_omics1, g_feature_omics1, feat_omics1, weight)
                output, alph = self.model(g_spatial_omics1, g_feature_omics1, feat_omics1, weight)
            total_loss = mse(output[train_mask], omics2_X[train_mask]).float()
            total_loss.backward()
            optimizer.step()
            
            vals.append(train_loss)
            if min(vals) != min(vals[-20:]):
                print('Early stopped.')
                break

        with th.no_grad():
            self.model.eval()

            if weight:
                output, wt, alph, latents = self.model(g_spatial_omics1, g_feature_omics1, feat_omics1, weight)
                if self.datatype=='Stereo-CITE-seq' or 'simu':
                    output = F.relu(output)
                train_loss = math.sqrt(mse(output[train_mask], omics2_X[train_mask]))

            else:
                output, alph = self.model(g_spatial_omics1, g_feature_omics1, feat_omics1, weight)
                if self.datatype=='Stereo-CITE-seq' or 'simu':
                    output = F.relu(output)
                train_loss = math.sqrt(mse(output[train_mask], omics2_X[train_mask]))

            adjust_learning_rate(optimizer, epoch, start_lr1=self.lr2, start_lr2=self.lr)


    

        y_pred = output.cpu()
        y_pred = y_pred.numpy()
        adata_omics2.obsm['predict'] = y_pred
        y_latens = latents.cpu()
        y_latens = y_latens.numpy()
        adata_omics2.obsm['latents'] = y_latens

        if os.path.exists(output_dir+'model'):
            th.save(self.model,output_dir+'model/sage_model.pth')
        else:
            os.makedirs(output_dir+'model')
            th.save(self.model,output_dir+'model/sage_model.pth')

        if save_csv:
            if self.type in ('ADT','RNA','spots'):
                y_pred = y_pred[test_idx]

            omics2_X = omics2_X[test_idx]
            omics2_X = omics2_X.cpu()
            omics2_X = omics2_X.numpy()
            df = pd.DataFrame(y_pred)
            df1 = pd.DataFrame(omics2_X)


            if os.path.exists(output_dir+'SpaMIE pred result'):
                # 将DataFrame保存为CSV文件
                df.to_csv(output_dir + 'SpaMIE pred result/' + str(self.seeds) + pred_name, index=False)
                df1.to_csv(output_dir + 'SpaMIE pred result/' + str(self.seeds) + true_name, index=False)
            else:
                os.makedirs(output_dir+'SpaMIE pred result')
                df.to_csv(output_dir + 'SpaMIE pred result/' + str(self.seeds) + pred_name, index=False)
                df1.to_csv(output_dir + 'SpaMIE pred result/' + str(self.seeds) + true_name, index=False)


        if weight:
            return adata_omics1, adata_omics2 , test_idx , train_idx , wt, alph
        else:
            return adata_omics1, adata_omics2 , test_idx , train_idx

                

        
