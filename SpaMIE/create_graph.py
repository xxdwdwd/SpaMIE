import scanpy as sc
import dgl
import torch as th
from SpaMIE.preprocess import preprocessing


def Sagegraph(modalities, device, datatype='Stereo-CITE-seq', batch=False):

    adata_omics1 = modalities[0]
    adata_omics2 = modalities[1]

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    # construct dgl graphs
    data = preprocessing(adata_omics1, adata_omics2, datatype=datatype, batch=batch)
    adata_omics1 = data['adata_omics1']
    adata_omics2 = data['adata_omics2']

    adj_spatial = adata_omics1.uns['adj_spatial']
    adj_feature = adata_omics1.obsm['adj_feature']
    x_spatial = list(adj_spatial['x'])
    y_spatial = list(adj_spatial['y'])
    adj_feature = adj_feature.tocoo()
    x_feature = list(adj_feature.row)
    y_feature = list(adj_feature.col)
    g_spatial_omics1 = dgl.graph((x_spatial, y_spatial))
    g_feature_omics1 = dgl.graph((x_feature, y_feature))
    g_spatial_omics1.ndata["feat"] = th.tensor(adata_omics1.obsm['feat'])
    g_feature_omics1.ndata["feat"] = th.tensor(adata_omics1.obsm['feat'])
    g_feature_omics1 = g_feature_omics1.to(device)
    g_spatial_omics1 = g_spatial_omics1.to(device)


    adj_spatial = adata_omics2.uns['adj_spatial']
    adj_feature = adata_omics2.obsm['adj_feature']
    x_spatial = list(adj_spatial['x'])
    y_spatial = list(adj_spatial['y'])
    adj_feature = adj_feature.tocoo()
    x_feature = list(adj_feature.row)
    y_feature = list(adj_feature.col)
    g_spatial_omics2 = dgl.graph((x_spatial, y_spatial))
    g_feature_omics2 = dgl.graph((x_feature, y_feature))
    g_spatial_omics2 = dgl.to_bidirected(g_spatial_omics2).to(device)
    g_feature_omics2 = dgl.to_bidirected(g_feature_omics2).to(device)
    return g_spatial_omics1, g_feature_omics1, g_spatial_omics2, g_feature_omics2, adata_omics1, adata_omics2


