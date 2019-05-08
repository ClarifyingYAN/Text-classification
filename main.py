# encoding=utf-8

from gensim.test.utils import common_texts, get_tmpfile
import gensim.utils as ut
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn import utils
import numpy as np
import multiprocessing
import jieba
import os
import subprocess


CWD = os.getcwd()  # 当前目录
corpus_file = 'toutiao_cat_data.txt'
data_dir = 'data'
raw_file = os.path.join(CWD, data_dir, corpus_file)  # 训练语料
model_name = 'doc2vec_model.d2v'
fname = get_tmpfile(os.path.join(CWD, data_dir, model_name))  # d2v 存储文件名
line_cnt = int(subprocess.check_output(['wc', '-l', raw_file]).strip().split()[0].decode("utf-8"))
test_quantity = line_cnt // 2
train_quantity = line_cnt - test_quantity

# 加载停用词
stopwords_list_name = os.path.join(CWD, "data", "ChineseStopwords.txt")
stopwords_list = []
with open(stopwords_list_name, 'r', encoding="utf-8") as f:
    for line in f:
        stopwords_list.append(line.rstrip())


# 中文分词预处理
def preprocessingCN(text):
    prepared_text = []
    split_text = list(jieba.cut(text, cut_all=False))
    for item in split_text:
        if item not in stopwords_list:
            prepared_text.append(item)
    return prepared_text


# 英文预处理
# def preprocessingEN(text):
#     return ut.simple_preprocess(text)


train_documents = []
test_documents = []
train_category = np.zeros(train_quantity)
test_category = np.zeros(test_quantity)
#
# tags_index = {'100': 0, '101': 1, '102': 2, '103': 3,
#               '104': 4, '105': 5, '106': 6, '107': 7,
#               '108': 8, '109': 9}
#
category = {}

# 读取语料至 TaggedDocument
with open(raw_file, 'r', encoding="utf-8") as f:
    i = 0
    for line in f:
        items = line.split('_!_')
        train_documents.append(TaggedDocument(
            words=preprocessingCN(items[3]),
            tags=[i]))
        if i < train_quantity:
            train_category[i] = items[1]
        else:
            test_category[i-train_quantity] = items[1]
        i += 1


# # gensim 生成训练生成向量
cores = multiprocessing.cpu_count()
vector_size = 100
model_dbow = Doc2Vec(dm=1, vector_size=vector_size, negative=5, hs=0, min_count=2,
                     sample=0, workers=cores, alpha=0.025, min_alpha=0.001)
model_dbow.build_vocab(train_documents)
train_documents = utils.shuffle(train_documents)
model_dbow.train(train_documents, total_examples=model_dbow.corpus_count,
                 epochs=30)

model_dbow.save(fname)

# model_dbow = Doc2Vec.load(fname)
train_arrays = np.zeros((train_quantity, vector_size))
test_arrays = np.zeros((test_quantity, vector_size))

for i in range(line_cnt):
    if i < train_quantity:
        train_arrays[i] = model_dbow[i]
    else:
        test_arrays[i-train_quantity] = model_dbow[i]

clf = LogisticRegression()
clf.fit(train_arrays, train_category)
clf.score(test_arrays, test_category)


# 100 民生 故事 news_story
# 101 文化 文化 news_culture
# 102 娱乐 娱乐 news_entertainment
# 103 体育 体育 news_sports
# 104 财经 财经 news_finance
# 106 房产 房产 news_house
# 107 汽车 汽车 news_car
# 108 教育 教育 news_edu
# 109 科技 科技 news_tech
# 110 军事 军事 news_military
# 112 旅游 旅游 news_travel
# 113 国际 国际 news_world
# 114 证券 股票 stock
# 115 农业 三农 news_agriculture
# 116 电竞 游戏 news_game