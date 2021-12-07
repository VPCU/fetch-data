import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stock-data-path', help='row stock data path', default='/home/mahaorui/gmoco/fetch-data/22mdata_c.pkl')
parser.add_argument('--gmoco-config-path', help='gmoco config yml path', default='/home/mahaorui/gmoco/fetch-data/log_train/2mnews/2021-11-29 20-35-29-pretrain/kd-news.yml')
parser.add_argument('--gmoco-model-path', help='gmoco model path', default='/home/mahaorui/gmoco/fetch-data/log_train/2mnews/2021-11-29 20-35-29-pretrain/pretrained_models/saved_encoder_2021-11-29 20-35-29.pkl')
parser.add_argument('--st-y', help='start year', default='2020')
parser.add_argument('--st-m', help='start month', default='1')
parser.add_argument('--ed-y', help='end year', default='2021')
parser.add_argument('--ed-m', help='end month', default='10')
parser.add_argument('--split-date', help='split date', default='2020-12-31')
parser.add_argument('--device', help='device', default='cuda')
parser.add_argument('--num-epochs', help='num epochs', default=50)
parser.add_argument('--learning-rate', help='learning rate', default=0.002)
parser.add_argument('--rnn-hidden-size', help='rnn hidden size', default=50)
parser.add_argument('--rnn-num-layers', help='rnn num layers', default=1)
parser.add_argument('--rnn-dropout', help='rnn dropout', default=0.2)
parser.add_argument('--rnn-fc-output-size', help='rnn fc output size', default=16)
parser.add_argument('--log-path', help='log path', default='.')
parser.add_argument('--news-row-path', help='news row path', default='/home/mahaorui/gmoco/fetch-data/data/news_row')
parser.add_argument('--news-feats-path', help='news feats path', default='/home/mahaorui/gmoco/fetch-data/data/news_embed')
parser.add_argument('--emb-model', help='emb model', default='gmoco')
parser.add_argument('--early_stopping_patience', help='early stopping patience', default=7)

args = parser.parse_args()

import sys
from os.path import join
import numpy as np
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from itertools import cycle
from model import NewsDataSet, TimeSeriesDataSet, RNN, add_gmoco_embeddings_to_stock_data, get_data_sets_list
import torch
from torch import nn
from sklearn import metrics
from pytorchtools import EarlyStopping


# Load stock data
df = pd.read_pickle(args.stock_data_path)
df['dt'] = pd.to_datetime(df['dt'])
start_y_m = f'{int(args.st_y)}-{int(args.st_m)}'
if int(args.ed_m) == 12:
    end_y_m = f'{int(args.ed_y) + 1}-01'
else:
    end_y_m = f'{int(args.ed_y)}-{int(args.ed_m) + 1}'
df = df[(df['dt'] >= start_y_m) & (df['dt'] < end_y_m)]
df['dt'] = df['dt'].apply(lambda x: x.date())
df = df.reset_index(drop=True)

# Add labels

df_open2 = df['open'].iloc[1:].values
df_open1 = df['close'].iloc[:-1].values

pre_label = df_open2 - df_open1
pre_label[pre_label >= 0] = 1
pre_label[pre_label < 0] = 0
pre_label = list(pre_label)
pre_label.append(0)
df['label'] = pre_label


# Normalize

nor_colomns = ['open', 'close', 'high', 'low', 'turnover', 'volume']
res_df = df[nor_colomns]
tmp_df = res_df.copy()

robustScaler = RobustScaler()
df_nor = robustScaler.fit_transform(res_df)
df_nor = pd.DataFrame(df_nor, columns=res_df.columns)

df_sub_columns = [c for c in df.columns if c not in df_nor.columns]
df_nor[df_sub_columns] = df[df_sub_columns]
df_nor = df_nor[df.columns]
df_nor = df_nor.fillna(0)

# Add gmoco embeddings

if args.emb_model == 'gmoco':
    data_sets = get_data_sets_list(int(args.st_y), int(args.st_m), int(args.ed_y), int(args.ed_m))
    loader = NewsDataSet(args.news_row_path, args.news_feats_path,  data_sets)
    df_nor = add_gmoco_embeddings_to_stock_data(df_nor, loader, args.gmoco_config_path, args.gmoco_model_path)


# Split training and test

df_nor.set_index(['dt'])
dateSepVal = pd.to_datetime(args.split_date)
dateSepVal = pd.Timestamp(dateSepVal)
train_df = df_nor[df_nor['dt'] <= dateSepVal]
test_df = df_nor[df_nor['dt'] > dateSepVal]

window_size = 5

# Data loader

train_dataset = TimeSeriesDataSet(train_df, window_size)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# test_loader = iter(cycle(DataLoader(TimeSeriesDataSet(test_df, window_size), batch_size=1, shuffle=False)))
test_loader = DataLoader(TimeSeriesDataSet(test_df, window_size), batch_size=1, shuffle=False)

# Train

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
num_epochs = args.num_epochs
learning_rate = args.learning_rate
input_size = train_dataset.feat_dim + 6
rnn = RNN(input_size, args.rnn_hidden_size, args.rnn_num_layers, args.rnn_fc_output_size, args.rnn_dropout)
criterion = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    # acc = correct_results_sum/y_test.shape[0]
    acc = correct_results_sum
    return acc.item()

rnn = rnn.to(device)

history = []

early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

for epoch in range(num_epochs):
    total_loss = 0
    total_test_loss = 0
    total_acc = 0
    total_test_acc = 0
    num_test = 0
    num_train = 0
    rnn.train()
    for (batch_idx, batch) in enumerate(train_loader):
        optimizer.zero_grad()
        bt_x_train, bt_y_train = batch
        bt_x_train = bt_x_train[0]
        bt_y_train = bt_y_train[0]
        num_train += bt_x_train.shape[0]
        x = bt_x_train.float()
        y = torch.Tensor(bt_y_train).float()
        y = y.view(-1, 1)
        x = x.to(device)
        y = y.to(device)
        output = rnn(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        acc = binary_acc(output, y)
        total_loss += loss.item()
        total_acc += acc
    rnn.eval()
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            bt_x_test, bt_y_test = batch
            bt_x_test = bt_x_test[0]
            bt_y_test = bt_y_test[0]
            num_test += bt_x_test.shape[0]
            x_test = bt_x_test.float()
            y_test = torch.Tensor(bt_y_test).float()
            y_test = y_test.view(-1, 1)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            output_test = rnn(x_test)
            test_loss = criterion(output_test, y_test)
            test_acc = binary_acc(output_test, y_test)
            total_test_loss += test_loss.item()
            total_test_acc += test_acc
    total_loss = total_loss / num_train
    total_test_loss = total_test_loss / num_test
    total_acc = total_acc / num_train
    total_test_acc = total_test_acc / num_test
    print("Epoch: %d, Loss: %f, Acc:%f, Test loss: %f, Test acc: %f" % (epoch+1, total_loss, total_acc, total_test_loss, total_test_acc))
    history.append([total_loss, total_acc, total_test_loss, total_test_acc])
    early_stopping(total_test_loss, rnn)
    if early_stopping.early_stop:
        print("Early stopping")
        break

rnn.load_state_dict(torch.load('checkpoint.pt'))


import matplotlib.pyplot as plt
loss = [x[0] for x in history]
acc = [x[1] for x in history]
test_loss = [x[2] for x in history]
test_acc = [x[3] for x in history]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, test_loss, 'b', label='Test loss')
plt.plot(epochs, test_acc, 'y', label='Test acc')  
plt.title('Loss and Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc/Loss')
plt.legend()
plt.savefig(join(args.log_path, 'loss_acc.png'))

# evaluate

eval_loader = DataLoader(TimeSeriesDataSet(test_df, window_size), batch_size=1, shuffle=False)

Y_test = []
Y_pred = []

rnn.eval()

with torch.no_grad():
    for (batch_idx, batch) in enumerate(eval_loader):
        bt_x_test, bt_y_test = batch
        bt_x_test = bt_x_test[0]
        bt_y_test = bt_y_test[0]
        x = bt_x_test.float()
        y = torch.Tensor(bt_y_test).float()
        y = y.view(len(y), 1)
        Y_test.append(y)
        x = x.to(device)
        y = y.to(device)
        output = rnn(x)
        Y_pred.append(output)

    Y_test = torch.cat(Y_test, dim=0)
    Y_pred = torch.cat(Y_pred, dim=0)

    Y_test = Y_test.cpu().detach().numpy()
    Y_pred = torch.sigmoid(Y_pred).cpu().detach().numpy()


draw=pd.concat([pd.DataFrame(Y_test),pd.DataFrame(Y_pred)],axis=1)
draw.iloc[:300,0].plot(figsize=(12,6))
draw.iloc[:300,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30')
plt.savefig(join(args.log_path, 'test_data.png'))

Y_pred_tag = Y_pred.round()

print('Accuracy :', metrics.accuracy_score(Y_test, Y_pred_tag))
print('F1-score :', metrics.f1_score(Y_test, Y_pred_tag))
