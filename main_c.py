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
train_corpus_file = 'cnews.train.txt'
test_corpus_file = 'cnews.test.txt'
data_dir = 'data'
train_raw_file = os.path.join(CWD, data_dir, train_corpus_file)  # 训练语料
test_raw_file = os.path.join(CWD, data_dir, test_corpus_file)  # 训练语料
model_name = 'doc2vec_model.d2v'
fname = get_tmpfile(os.path.join(CWD, data_dir, model_name))  # d2v 存储文件名
train_line_cnt = int(subprocess.check_output(['wc', '-l', train_raw_file]).strip().split()[0].decode("utf-8"))
test_line_cnt = int(subprocess.check_output(['wc', '-l', test_raw_file]).strip().split()[0].decode("utf-8"))
test_quantity = test_line_cnt
train_quantity = train_line_cnt

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

tags_index = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3,
              '教育': 4, '时尚': 5, '时政': 6, '游戏': 7,
              '科技': 8, '财经': 9}

# category = {}

# 读取语料至 TaggedDocument
print('Start to read train file:')
with open(train_raw_file, 'r', encoding="utf-8") as f:
    i = 0
    for line in f:
        print('read {}'.format(i))
        items = line.split('\t')
        train_documents.append(TaggedDocument(
            words=preprocessingCN(items[1]),
            tags=[i]))
        train_category[i] = tags_index.get(items[0])
        i += 1

print('Start to read test file:')
with open(test_raw_file, 'r', encoding="utf-8") as f:
    i = 0
    for line in f:
        print('read {}'.format(i))
        items = line.split('\t')
        test_documents.append(TaggedDocument(
            words=preprocessingCN(items[1]),
            tags=[i+train_line_cnt]))
        test_category[i] = tags_index.get(items[0])
        i += 1

all_documents = [*train_documents, *test_documents]

# # gensim 生成训练生成向量
print('Start to doc2vec:')
cores = multiprocessing.cpu_count()
vector_size = 100
model_dbow = Doc2Vec(dm=1, vector_size=vector_size, negative=5, hs=0, min_count=2,
                     sample=0, workers=cores, alpha=0.025, min_alpha=0.001)
model_dbow.build_vocab(all_documents)
all_documents = utils.shuffle(all_documents)
model_dbow.train(all_documents, total_examples=model_dbow.corpus_count,
                 epochs=30)

model_dbow.save(fname)
print('Save model!')

# model_dbow = Doc2Vec.load(fname)
train_arrays = np.zeros((train_quantity, vector_size))
test_arrays = np.zeros((test_quantity, vector_size))

for i in range(train_line_cnt+test_line_cnt):
    if i < train_line_cnt:
        train_arrays[i] = model_dbow[i]
    else:
        test_arrays[i-train_line_cnt] = model_dbow[i]

print('LR classification:')
clf = LogisticRegression()
clf.fit(train_arrays, train_category)
print('LR fit finished!')
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