import pandas as pd
import scipy as sc
import networkx as nx
import numpy as np
import json
import os
from tqdm import tqdm
from os.path import join


def text_wrap(text):
    return text.replace('######', '') + '  '


def rel_orgs_wrap(text):
    return text.split('|') if text else []


class NewsDataSet:
    def __init__(self, row_path, feat_path, data_path_list):
        self.row_path = row_path
        self.feat_path = feat_path
        self.data_path_list = data_path_list
        self.index = 0
        fst_feat_path = join(self.feat_path, self.data_path_list[0]+'.npy')
        fst_feat = np.load(fst_feat_path, allow_pickle=True)
        self.feat_dim = fst_feat.shape[1]
        # if len(self.X) != len(self.Y):
        #  raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.data_path_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data_path_list):
            raise StopIteration
        index = self.index
        self.index += 1
        row_path = join(self.row_path, self.data_path_list[index]+'.pkl')
        feat_path = join(self.feat_path, self.data_path_list[index]+'.npy')
        df = pd.read_pickle(row_path)
        feat = np.load(feat_path, allow_pickle=True)
        _df = pd.DataFrame()
        _df['id'] = df['id']
        _df['rel_orgs'] = df['rel_org_a_companies_code'].map(rel_orgs_wrap)
        _df['date'] = df['date_time'].apply(lambda x: x.date())
        _df['feat'] = _df.apply(lambda x: feat[x.name], axis=1)
        df = _df
        return df


def build_graph(data_loader):
    G = nx.Graph()
    feat_dim = data_loader.feat_dim

    def to_feat(k):
        if isinstance(k, int):
            # Document node
            return G.nodes[k]['feat']
        else:
            # Stock node
            return np.zeros(feat_dim)
    for df_month in tqdm(data_loader):
        for _, node in df_month.iterrows():
            for org in node['rel_orgs']:
                G.add_edge(node.id, org)
                G.nodes[node.id]['feat'] = node['feat']
    ssm = nx.convert_matrix.to_scipy_sparse_matrix(G)
    feats = map(to_feat, G.nodes)
    feats = np.stack(list(feats))
    return ssm, feats


def save_graph(ssm, feats, data_prefix):
    os.path.exists(data_prefix) or os.makedirs(data_prefix)
    sc.sparse.save_npz(join(data_prefix, 'adj_full.npz'), ssm)
    sc.sparse.save_npz(join(data_prefix, 'adj_train.npz'), ssm)
    np.save(join(data_prefix, 'feats.npy'), feats)
    number_of_nodes = feats.shape[0]
    json.dump(dict.fromkeys(range(number_of_nodes), 0),
              open(join(data_prefix, 'class_map.json'), 'w'))
    role = {'tr': list(range(number_of_nodes)), 'va': [], 'te': []}
    json.dump(role, open(join(data_prefix, 'role.json'), 'w'))
