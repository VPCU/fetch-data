import os
os.environ['kdconfig']='../../database.yaml'
import argparse
import pandas as pd
import jieba
import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-y', help='year')
parser.add_argument('-m', help='month')
args = parser.parse_args()

y = int(args.y)
m = int(args.m)


df = pd.read_pickle(f'../news_row/{y}-{m}.pkl')

def text_wrap(text):
    if not text:
        text = ''
    return text.replace('######','').replace('<p>', '').replace('</p>', '').replace('&nbsp;', '') + '  '

rs = pd.DataFrame()
rs['title'] = df['title']
rs['text'] = df['summary'].map(text_wrap)+df['content'].map(text_wrap)

def cut(t):
    # stopwords = [line[:-1] for line in open('stopwords.txt', encoding='utf-8')]
    result = []
    for i in t:
        i_c = jieba.cut(i)
        i_c = list(i_c)
        # i_c = [i for i in i_c if i not in stopwords]
        result.append(i_c)
    return result

def prepare_doc2vec(data):
    result = []
    for i, t in enumerate(data):
        l = len(t)
        if l == 0:
            continue
        t[l-1] = t[l-1].strip()
        documents = TaggedDocument(t, [i])
        result.append(documents)
    return result

def train(data, size=300):
    model = Doc2Vec(data, window=5, vector_size=size, min_count=1, workers=12)
    model.train(data, total_examples=model.corpus_count, epochs=10)
    return model

model = train(prepare_doc2vec(cut(rs['title'])))
np.save(f'{y}-{m}.npy', model.dv.vectors)