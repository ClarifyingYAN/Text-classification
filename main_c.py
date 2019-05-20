# encoding=utf-8

from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn import svm, utils, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import multiprocessing
import jieba
import os
import subprocess
import timeit
import re


CWD = os.getcwd()  # 当前目录
train_corpus_file = 'cnews.train.txt'
test_corpus_file = 'cnews.test.txt'
data_dir = 'data'
train_raw_file = os.path.join(CWD, data_dir, train_corpus_file)  # 训练语料
test_raw_file = os.path.join(CWD, data_dir, test_corpus_file)  # 训练语料
train_line_cnt = int(subprocess.check_output(['wc', '-l', train_raw_file])
                     .strip().split()[0].decode("utf-8"))
test_line_cnt = int(subprocess.check_output(['wc', '-l', test_raw_file]).strip()
                    .split()[0].decode("utf-8"))
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
        if not re.findall(r'[·0-9a-zA-Z\s\\%]', item.replace('\n', '0').replace('\xa0', '1')) and item not in stopwords_list:
            prepared_text.append(item)
    return prepared_text


train_documents = []
test_documents = []
train_category = np.zeros(train_quantity)
test_category = np.zeros(test_quantity)

tags_index = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3,
              '教育': 4, '时尚': 5, '时政': 6, '游戏': 7,
              '科技': 8, '财经': 9}
inverse_tags_index = dict([val, key] for key, val in tags_index.items())
train_tag_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
test_tag_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# category = {}

# 读取语料至 TaggedDocument
print('Start to read train file:')
start = timeit.default_timer()
with open(train_raw_file, 'r', encoding="utf-8") as f:
    i = 0
    for line in f:
        print('read {}'.format(i))
        items = line.split('\t')
        train_documents.append(TaggedDocument(
            words=preprocessingCN(items[1]),
            tags=[i]))
        train_category[i] = tags_index.get(items[0])
        train_tag_cnt[tags_index.get(items[0])] += 1
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
        test_tag_cnt[tags_index.get(items[0])] += 1
        i += 1

stop = timeit.default_timer()
print('time for loading file: {} seconds'.format(stop - start))
for i in range(10):
    print('{0} {1} data for training, {2} {1} for testing'
          .format(train_tag_cnt[i], inverse_tags_index.get(i), test_tag_cnt[i]))
all_documents = [*train_documents, *test_documents]

# 打印观察
print(train_documents[100])


def doc2vec_training(documents, v_size, dm=1):
    print('Start to doc2vec: v_size={}, dm={}, document_size={}'
          .format(v_size, dm, len(documents)))
    cores = multiprocessing.cpu_count()
    model = Doc2Vec(dm=dm, vector_size=v_size, negative=5, hs=0, min_count=1,
                    sample=0, workers=cores, alpha=0.025, min_alpha=0.001)
    begin = timeit.default_timer()
    model.build_vocab(documents)
    end = timeit.default_timer()
    print('time for doc2vec to build vocabulary: {} seconds'
          .format(end - begin))
    documents = utils.shuffle(documents)
    begin = timeit.default_timer()
    model.train(documents, total_examples=model.corpus_count, epochs=30)
    end = timeit.default_timer()
    print('time for doc2vec model training: {} seconds'.format(end - begin))

    model_name = '_'.join(['doc2vec', 'vsize={}'.format(v_size),
                          'dm={}'.format(dm), 'model.d2v'])
    fname = get_tmpfile(os.path.join(CWD, data_dir, model_name))  # d2v 存储文件名
    model.save(fname)
    print('Save model!')
    return model

# # gensim 生成训练生成向量
# print('Start to doc2vec:')
# cores = multiprocessing.cpu_count()
# vector_size = 100
# vector_size2 = 200
# vector_size3 = 300
# model_dbow = Doc2Vec(dm=1, vector_size=vector_size, negative=5, hs=0,
#                      min_count=1, sample=0, workers=cores, alpha=0.025,
#                      min_alpha=0.001)
# model_dbow.build_vocab(all_documents)
# all_documents = utils.shuffle(all_documents)
# start = timeit.default_timer()
# model_dbow.train(all_documents, total_examples=model_dbow.corpus_count,
#                  epochs=30)
# stop = timeit.default_timer()
# print('time for doc2vec model training: {} seconds'.format(stop - start))
#
# model_dbow.save(fname)
# print('Save model!')

# model_dbow = Doc2Vec.load(fname)


def get_vectors(model, v_size, quantity):
    train_x = np.zeros((quantity, v_size))
    test_x = np.zeros((test_quantity, v_size))

    for j in range(quantity + test_line_cnt):
        if j < quantity:
            train_x[j] = model[j]
        else:
            test_x[j - quantity] = model[j]

    return train_x, test_x


def ML_training(clf_name, clf_instance, train_x, train_y, test_x, test_y):
    print('===================================================================')
    print('{} classification:'.format(clf_name))
    clf = clf_instance
    begin = timeit.default_timer()
    clf.fit(train_x, train_y)
    end = timeit.default_timer()
    print('{} fit finished!'.format(clf_name))
    print('time for {} model machine learning: {} seconds'
          .format(clf_name, end - begin))
    clf.score = clf.score(test_x, test_y)
    print('{} Score is  {}'.format(clf_name, clf.score))


def ML_all(train_x, train_y, test_x, test_y):
    LR_clf = LogisticRegression()
    ML_training('LogisticRegression', LR_clf, train_x, train_y,
                test_x, test_y)

    SVC_clf = svm.SVC()
    ML_training('SVC', SVC_clf, train_x, train_y,
                test_x, test_y)

    T_clf = tree.DecisionTreeClassifier()
    ML_training('DecisionTree', T_clf, train_x, train_y,
                test_x, test_y)

    RF_clf = RandomForestClassifier()
    ML_training('RandomForest', RF_clf, train_x, train_y,
                test_x, test_y)

    SGD_clf = SGDClassifier()
    ML_training('SGD', SGD_clf, train_x, train_y,
                test_x, test_y)


trained_model = doc2vec_training(documents=all_documents, v_size=100, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=100, quantity=50000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=all_documents, v_size=200, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=200, quantity=50000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=all_documents, v_size=300, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=300, quantity=50000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=all_documents, v_size=100, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=100, quantity=50000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=all_documents, v_size=200, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=200, quantity=50000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=all_documents, v_size=300, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=300, quantity=50000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

s_documents = [*train_documents[0:1000], *train_documents[5000:6000],
               *train_documents[10000:11000], *train_documents[15000:16000],
               *train_documents[20000:21000], *train_documents[25000:26000],
               *train_documents[30000:31000], *train_documents[35000:36000],
               *train_documents[40000:41000], *train_documents[45000:46000],
               *test_documents]
s_category = [*train_category[0:1000], *train_category[5000:6000],
              *train_category[10000:11000], *train_category[15000:16000],
              *train_category[20000:21000], *train_category[25000:26000],
              *train_category[30000:31000], *train_category[35000:36000],
              *train_category[40000:41000], *train_category[45000:46000]]
trained_model = doc2vec_training(documents=s_documents, v_size=100, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=100, quantity=10000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=s_documents, v_size=200, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=200, quantity=10000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=s_documents, v_size=300, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=300, quantity=10000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=s_documents, v_size=100, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=100, quantity=10000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=s_documents, v_size=200, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=200, quantity=10000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=s_documents, v_size=300, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=300, quantity=10000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

m_documents = [*train_documents[0:2500], *train_documents[5000:7500],
               *train_documents[10000:12500], *train_documents[15000:17500],
               *train_documents[20000:22500], *train_documents[25000:27500],
               *train_documents[30000:32500], *train_documents[35000:37500],
               *train_documents[40000:42500], *train_documents[45000:47500],
               *test_documents]
m_category = [*train_category[0:2500], *train_category[5000:7500],
              *train_category[10000:12500], *train_category[15000:17500],
              *train_category[20000:22500], *train_category[25000:27500],
              *train_category[30000:32500], *train_category[35000:37500],
              *train_category[40000:42500], *train_category[45000:47500]]
trained_model = doc2vec_training(documents=m_documents, v_size=100, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=100, quantity=25000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=m_documents, v_size=200, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=200, quantity=25000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=m_documents, v_size=300, dm=1)
train_arrays, test_arrays = get_vectors(trained_model, v_size=300, quantity=25000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=m_documents, v_size=100, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=100, quantity=25000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=m_documents, v_size=200, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=200, quantity=25000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

trained_model = doc2vec_training(documents=m_documents, v_size=300, dm=0)
train_arrays, test_arrays = get_vectors(trained_model, v_size=300, quantity=25000)
ML_all(train_arrays, train_category, test_arrays, test_category)
print(trained_model[100])

