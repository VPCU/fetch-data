import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import torch.nn as nn
import torch
import networkx as nx
import os
import sys
sys.path.append('/home/mahaorui/gmoco/moco-test')
from graphsaint.mocodict.encoder_predict import encoder_predict

class TimeSeriesDataSet(Dataset):
    def _flat_embs(self, b):
        embs = b[:, -2]
        embs = list(map(lambda x: np.zeros(self.feat_dim)
                    if x is None else x, embs))
        embs = np.array(embs)
        return np.hstack((b[:, :-2], embs, b[:, -1:]))

    def __init__(self, data, window_size):
        df_grouped = [x for _, x in data.groupby(
            'kdcode') if len(x) > window_size]
        self.df_grouped = df_grouped
        self.window_size = window_size
        self.sequence_length = window_size + 1
        if 'emb' not in data.columns:
            self.feat_dim = 0
        else:
            for i in data['emb']:
                if i is not None:
                    self.feat_dim = i.shape[0]
                    break

    def __len__(self):
        return len(self.df_grouped)

    def __getitem__(self, index):
        data = []
        stock = self.df_grouped[index]
        stock = np.array(stock)
        if self.feat_dim > 0:
            stock = self._flat_embs(stock)
        for i in range(len(stock) - self.window_size):
            data.append(stock[i:i + self.sequence_length])

        data = np.array(data)
        X = data[:, :-1, 2:-1]
        Y = data[:, -1, -1]

        X = X.astype('float32')
        Y = Y.astype('float32')

        return X, Y


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fc_output_size, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size,
                             num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, fc_output_size)
        self.activation = nn.ReLU()
        self.fc_1 = nn.Linear(fc_output_size, 1)

    def forward(self, input):
        lstm_out, _ = self.lstm1(input)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.activation(out)
        out = self.fc_1(out)
        return out




def rel_orgs_wrap(text):
    return text.split('|') if text else []


class NewsDataSet:
  def __init__(self, row_path, feat_path, data_path_list):
    self.row_path = row_path
    self.feat_path = feat_path
    self.data_path_list = data_path_list
    self.index = 0
    #if len(self.X) != len(self.Y):
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
    row_path = os.path.join(self.row_path, self.data_path_list[index]+'.pkl')
    feat_path = os.path.join(self.feat_path, self.data_path_list[index]+'.npy')
    df = pd.read_pickle(row_path)
    feat = np.load(feat_path, allow_pickle=True)
    _df = pd.DataFrame()
    _df['id'] = df['id']
    _df['rel_orgs'] = df['rel_org_a_companies_code'].map(rel_orgs_wrap)
    _df['date'] = df['date_time'].apply(lambda x: x.date())
    _df['feat'] = _df.apply(lambda x: feat[x.name], axis=1)
    df = _df
    return df


class TheGraph(nx.Graph):
    def doc_nodes(self):
        return [n for n in self.nodes if not isinstance(n, str)]
    def stock_nodes(self):
        return [n for n in self.nodes if isinstance(n, str)]
    def get_feats(self):
        return np.array([self.nodes[k]['feat'] for k in self.nodes])
    def add_doc_node(self, node_id, feat, adj_to=[], days=7):
        self.add_node(node_id, feat=feat, valid_days=days)
        for stock in adj_to:
            if stock not in self.nodes:
                self.add_node(stock, feat=np.zeros(feat.shape[0]))
            self.add_edge(node_id, stock)
    def get_adj_matrix(self):
        return nx.convert_matrix.to_scipy_sparse_matrix(self)
    def new_day(self):
        for node in self.doc_nodes():
            self.nodes[node]['valid_days'] -= 1
    def remove_outdated_nodes(self):
        for node in self.doc_nodes():
            if self.nodes[node]['valid_days'] <= 0:
                self.remove_node(node)
    def remove_isolated_nodes(self):
        self.remove_nodes_from(list(nx.isolates(self)))
    def clean_nodes(self):
        self.remove_outdated_nodes()
        self.remove_isolated_nodes()

def get_data_sets_list(st_y, st_m, ed_y, ed_m):
    data_sets_list = []
    st = st_y * 12 + (st_m - 1)
    ed = ed_y * 12 + (ed_m - 1)
    for i in range(st, ed + 1):
        y = i // 12
        m = i % 12 + 1
        data_sets_list.append(f'{y}-{m}')
    return data_sets_list


def add_gmoco_embeddings_to_stock_data(df_nor, news_loader, config_path, model_path):
    df_nor['idx'] = df_nor['kdcode'] + ',' + df_nor['dt'].apply(str)
    df_nor.set_index('idx', inplace=True)
    df_nor['emb'] = None
    df_nor['emb'] = df_nor['emb'].astype(object)

    
    loader = news_loader
    G = TheGraph()

    for df_month in loader:
        doc_df_grouped = {k:v for k, v in df_month.groupby('date')}
        # for date, docs in tqdm(doc_df_grouped.items()):
        for date, docs in doc_df_grouped.items():
            G.new_day()
            for _, doc in docs.iterrows():
                G.add_doc_node(doc.id, doc.feat, doc.rel_orgs, days=14)
            G.clean_nodes()
            feats = G.get_feats()
            adj = G.get_adj_matrix()
            output = encoder_predict(adj, feats, model_path, config_path).detach()
            date_ = str(date)
            for kdnode, emb in zip(G.nodes, output):
                if isinstance(kdnode, str):
                    k = kdnode+','+date_
                    if k in df_nor.index:
                        df_nor.at[kdnode+','+date_, 'emb'] = emb.numpy()


    df_nor_grouped = [ x for _, x in df_nor.groupby('kdcode') ]
    for s in df_nor_grouped:
        s['emb'] = s['emb'].fillna(method='pad', limit=7)
    df_nor = pd.concat(df_nor_grouped)
    del df_nor_grouped

    df_nor.reset_index(drop=True)
    _l = list(df_nor.columns)
    _l.remove('label')
    _l.append('label')
    df_nor = df_nor[_l]
    return df_nor
