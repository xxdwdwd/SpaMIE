
import math
import random
import dgl
import dgl.nn.pytorch as dglnn
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def propagation_layer_combination(X, wt, from_logits=True):

    if from_logits:
        wt = th.softmax(wt, -1)


    x = 0
    for i in range(wt.shape[0]):
        x += wt[i] * X[i]

    return x, wt


def propagation_layer_combination_new(X, wt, from_logits=True):

    if from_logits:
        wt = torch.stack(wt)
        wt = th.softmax(wt, -1)


    x = 0
    for i in range(wt.shape[0]):
        x += wt[i] * X[i]

    return x, wt

class AttentionLayer_between_modality(Module):
    """
    Attention layer
    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer_between_modality, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        # emb.append(torch.unsqueeze(torch.squeeze(emb3), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha


class AttentionLayer_within_modality(Module):
    """
    Attention layer
    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer_within_modality, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha


class cir_mlp(Module):
    def __init__(self, n_hidden, out_feats, dropout=0.0):
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.batch1 = nn.BatchNorm1d(1024)
        self.batch2 = nn.BatchNorm1d(n_hidden)
        self.linear1_1 = nn.Linear(n_hidden, 1024)
        self.linear2_2 = nn.Linear(1024, n_hidden)
        self.linear3_3 = nn.Linear(n_hidden, out_feats)

    def forward(self, x):
        x = self.linear1_1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2_2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear3_3(x)
        x = F.relu(x)
        return x


class SpaMIE_pred(nn.Module):
    def __init__(self,
                    in_feats,
                    n_hidden,
                    out_feats,
                    wt,
                    activation,
                    sagetype,
                    layers_nums,  
                    res, 
                    batchnorm,                 
                    dropout):
        super().__init__()

        self.layers_num = layers_nums
        self.activation = activation
        self.n_hidden = n_hidden
        self.res = res
        self.batchnorm = batchnorm
        self.wt = wt

        self.conv_layers = nn.ModuleList()
        self.conv_acts = nn.ModuleList()


        for i in range(self.layers_num):
            if i==0:
                self.conv_layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            else:
                self.conv_layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        
        if self.activation is not None:
            for i in range(self.layers_num):
                self.conv_acts.append(self.activation())

        self.linear1 = nn.Linear(n_hidden, 1024)
        self.linear2 = nn.Linear(1024, n_hidden)
        self.linear3 = nn.Linear(n_hidden, out_feats)
        self.batch1 = nn.BatchNorm1d(1024)
        self.batch2 = nn.BatchNorm1d(n_hidden)
        self.batch3 = nn.BatchNorm1d(n_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.atten_omics1 = AttentionLayer_within_modality(n_hidden, n_hidden)
        self.wt1 = Parameter(th.FloatTensor(wt))
        self.wt2 = Parameter(th.FloatTensor(wt))

    def encoder(self, g, feat):
        hcell = []
        x = self.conv_layers[0](g, feat)
        if self.batchnorm:
            x = self.batch3(x)
        if self.activation is not None:
            x = self.conv_acts[0](x)
        
        hcell.append(x)

        for i in range(self.layers_num-1):
            x = self.dropout1(x)
            
            if self.res == 'res_add':
                x = self.conv_layers[i+1](g, x) + x
            else:
                x = self.conv_layers[i+1](g, x)
            if self.batchnorm:
                x = self.batch3(x)
            if self.activation is not None:
                x = self.conv_acts[i+1](x)
            hcell.append(x)
            
        return hcell

    def read_out(self, x):
        x = self.linear1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear3(x)
        return x
    

    def forward(self, g_spatial_omics1, g_feature_omics1, feat_omics1, weight):
        # omics1 to omics2
        x_spatial_omics1 = self.encoder(g_spatial_omics1, feat_omics1)
        x_feature_omics1 = self.encoder(g_feature_omics1, feat_omics1)

        if weight:
            emb_latent_spatial, wt_spa = propagation_layer_combination(x_spatial_omics1, self.wt1)
            emb_latent_feature, wt_fea = propagation_layer_combination(x_feature_omics1, self.wt2)
            emb_latent_omics1, alph = self.atten_omics1(emb_latent_spatial, emb_latent_feature)
            emb_map_omics1 = self.read_out(emb_latent_omics1)
            return emb_map_omics1, [wt_spa, wt_fea], alph , emb_latent_omics1
        else:
            emb_latent_omics1,alph = self.atten_omics1(x_spatial_omics1[-1], x_feature_omics1[-1])
            emb_latent_omics1 = self.read_out(emb_latent_omics1)
            return emb_latent_omics1, alph

class SpaMIE_joint(nn.Module):
    def __init__(self,
                    in_feats,
                    n_hidden,
                    out_feats,
                    layers_nums,
                    dropout,
                    wt=[1.0]*3,
                    activation=None,    
                    res=True                            
                    ):
        super().__init__()

        self.layers_num = layers_nums
        self.activation = activation
        self.n_hidden = n_hidden
        self.res = res
        self.wt = wt

        self.conv_layers_omics1 = nn.ModuleList()
        self.conv_layers_omics2 = nn.ModuleList()

        self.decoder_layers_omics1 = nn.ModuleList()
        self.decoder_layers_omics2 = nn.ModuleList()

        self.conv_acts = nn.ModuleList()


        for i in range(self.layers_num):
            if i==0:
                self.conv_layers_omics1.append(dglnn.SAGEConv(in_feats, n_hidden, 'gcn'))
            else:
                self.conv_layers_omics1.append(dglnn.SAGEConv(n_hidden, n_hidden, 'gcn'))
        
        for i in range(self.layers_num):
            if i==0:
                self.decoder_layers_omics1.append(dglnn.SAGEConv(n_hidden, in_feats, 'gcn'))
            else:
                self.decoder_layers_omics1.append(dglnn.SAGEConv(in_feats, in_feats, 'gcn'))
        
        for i in range(self.layers_num):
            if i==0:
                self.decoder_layers_omics2.append(dglnn.SAGEConv(n_hidden, in_feats, 'gcn'))
            else:
                self.decoder_layers_omics2.append(dglnn.SAGEConv(in_feats, in_feats, 'gcn'))

        for i in range(self.layers_num):
            if i==0:
                self.conv_layers_omics2.append(dglnn.SAGEConv(in_feats, n_hidden, 'gcn'))
            else:
                self.conv_layers_omics2.append(dglnn.SAGEConv(n_hidden, n_hidden, 'gcn'))
     
        if self.activation is not None:
            for i in range(self.layers_num):
                self.conv_acts.append(self.activation())

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.atten_omics1 =  AttentionLayer_within_modality(n_hidden, n_hidden)
        self.atten_omics2 =  AttentionLayer_within_modality(n_hidden, n_hidden)
        self.atten_cross = AttentionLayer_between_modality(n_hidden, n_hidden)
        self.wt1 = Parameter(th.FloatTensor(self.wt))
        self.wt2 = Parameter(th.FloatTensor(self.wt))
        self.wt3 = Parameter(th.FloatTensor(self.wt))
        self.wt4 = Parameter(th.FloatTensor(self.wt))

    def encoder_omics1(self, g, feat):
        hcell = []
        x = self.conv_layers_omics1[0](g, feat)
        
        if self.activation is not None:
            x = self.conv_acts[0](x)

        hcell.append(x)

        for i in range(self.layers_num-1):
            x = self.dropout1(x)
            
            if self.res == 'res_add':
                x = self.conv_layers_omics1[i+1](g, x) + x
            else:
                x = self.conv_layers_omics1[i+1](g, x)

            if self.activation is not None:
                x = self.conv_acts[i+1](x)
            hcell.append(x)
            
        return hcell

    def encoder_omics2(self, g, feat):
        hcell = []
        x = self.conv_layers_omics2[0](g, feat)
        
        if self.activation is not None:
            x = self.conv_acts[0](x)
        hcell.append(x)

        for i in range(self.layers_num-1):
            x = self.dropout1(x)
            
            if self.res == 'res_add':
                x = self.conv_layers_omics2[i+1](g, x) + x
            else:
                x = self.conv_layers_omics2[i+1](g, x)

            if self.activation is not None:
                x = self.conv_acts[i+1](x)
            hcell.append(x)
            
        return hcell

    def decoder_omics1(self, g, feat):
        x = self.decoder_layers_omics1[0](g, feat)
        
        if self.activation is not None:
            x = self.conv_acts[0](x)


        for i in range(self.layers_num-1):
            x = self.dropout1(x)
            
            if self.res == 'res_add':
                x = self.decoder_layers_omics1[i+1](g, x) + x
            else:
                x = self.decoder_layers_omics1[i+1](g, x)

            if self.activation is not None:
                x = self.conv_acts[i+1](x)

        return x

    def decoder_omics2(self, g, feat):
        x = self.decoder_layers_omics2[0](g, feat)
        
        if self.activation is not None:
            x = self.conv_acts[0](x)


        for i in range(self.layers_num-1):
            x = self.dropout1(x)
            
            if self.res == 'res_add':
                x = self.decoder_layers_omics2[i+1](g, x) + x
            else:
                x = self.decoder_layers_omics2[i+1](g, x)

            if self.activation is not None:
                x = self.conv_acts[i+1](x)

        return x


    def forward(self,  g_spatial_omics1, g_feature_omics1, feat_omics1,
                g_spatial_omics2, g_feature_omics2, feat_omics2, weight=True):
        # omics1 encoder
        x_spatial_omics1 = self.encoder_omics1(g_spatial_omics1, feat_omics1)
        x_feature_omics1 = self.encoder_omics1(g_feature_omics1, feat_omics1)

        if weight:
            emb_latent_spatial_omics1, wt_spa_omics1 = propagation_layer_combination(x_spatial_omics1, self.wt1)
            emb_latent_feature_omics1, wt_fea_omics1 = propagation_layer_combination(x_feature_omics1, self.wt2)
            emb_latent_omics1, alph1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        else:
            emb_latent_omics1, alph1 = self.atten_omics1(x_spatial_omics1[-1], x_feature_omics1[-1])

        # omics2 encoder
        x_spatial_omics2 = self.encoder_omics2(g_spatial_omics2, feat_omics2)
        x_feature_omics2 = self.encoder_omics2(g_feature_omics2, feat_omics2)
        if weight:
            emb_latent_spatial_omics2, wt_spa_omics2 = propagation_layer_combination(x_spatial_omics2, self.wt3)
            emb_latent_feature_omics2, wt_fea_omics2 = propagation_layer_combination(x_feature_omics2, self.wt4)
            emb_latent_omics2, alph2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        else:
            emb_latent_omics2, alph2 = self.atten_omics2(x_spatial_omics2[-1], x_feature_omics2[-1])
        
        # with between-modality attention aggregation layer
        emb_latent_combined, alph = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        # omics2 decoder
        dec_latent_omics1 = self.decoder_omics1(g_spatial_omics1, emb_latent_combined)
        dec_latent_omics2 = self.decoder_omics2(g_spatial_omics2, emb_latent_combined)

        # cross modality decoder

        cross_latent_omics1 = self.encoder_omics2(g_spatial_omics1, self.decoder_omics2(g_spatial_omics1, emb_latent_omics1))
        cross_latent_omics2 = self.encoder_omics1(g_spatial_omics2, self.decoder_omics1(g_spatial_omics2, emb_latent_omics2))

        if weight:
            wt = [wt_spa_omics1, wt_fea_omics1, wt_spa_omics2, wt_fea_omics2]
            return (emb_latent_combined, wt, dec_latent_omics1, dec_latent_omics2,
                 emb_latent_omics1, emb_latent_omics2,
                 cross_latent_omics1, cross_latent_omics2, alph)  
        else:
            return (emb_latent_combined, dec_latent_omics1, dec_latent_omics2,
                    emb_latent_omics1, emb_latent_omics2,
                    cross_latent_omics1, cross_latent_omics2, alph)

    