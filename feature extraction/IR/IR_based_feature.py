# -*-coding:utf-8-*-
# @Time : 2022/10/10 20:44
# @Author : 邓洋、李幸阜
import scipy
from gensim import corpora, similarities
from gensim import models
import re
import numpy as np
import pandas as pd
import scipy.stats
import os
from models.rank_bm25 import BM25Okapi
import matplotlib.pyplot as plt
import itertools
from gensim.models.word2vec import Word2Vec
from gensim.similarities import WmdSimilarity
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings(action='ignore')

# 生成查询集或被查询集(生成数据集)
def set_generation(query_file):
    """
    生成查询集或被查询集(生成数据集)
    :param query_file: 分词和去除停用词后的数据集
    :return: 返回一个列表，列表的每个元素也是一个列表，后者中的列表的每个元素都是每一条数据中的单词。
    """
    with open(query_file, "r", encoding="gbk") as ft:
        lines_T = ft.readlines()
    setline = []
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        word = [i for i in word if len(i) > 0]
        setline.append(word)
    return setline


# VSM相似度计算
def vsm_similarity(queried_file, query_file, output_fname=None):
    # 生成被查询集
    queried_line = set_generation(queried_file)
    # 生成查询集
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_tfidf)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    sim = pd.DataFrame(corpus_sim[query_tfidf])
    print(sim)
    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim


# LSI相似度计算
def lsi_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 生成lsi主题
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary)
    corpus_lsi = lsi_model[corpus_tfidf]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_lsi)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    query_lsi = lsi_model[query_tfidf]
    sim = pd.DataFrame(corpus_sim[query_lsi])

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim


# LDA相似度计算
def lda_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    # tfidf_model = models.TfidfModel(corpus)
    # corpus_tfidf = tfidf_model[corpus]

    # 生成lda主题
    topic_number = 100
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_number, random_state=0)
    document_topic = lda_model.get_document_topics(corpus)
    # corpus_lda = lda_model[corpus_tfidf]

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    # query_tfidf = tfidf_model[query_corpus]
    query_lda = lda_model.get_document_topics(query_corpus)

    sim = hellingerSim(document_topic, query_lda, topic_number)

    if output_fname is not None:
        sim.to_excel(output_fname)
    print(sim)
    return sim


# BM25分数计算
def bm25_score(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    bm25 = BM25Okapi(queried_line)
    scores = pd.DataFrame(bm25.get_full_scores(query_line))
    if output_fname is not None:
        scores.to_excel(output_fname)
    return scores



# JS散度相似度计算
def JS_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line+query_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]
    corpus2 = [dictionary.doc2bow(text) for text in query_line]
    A_matrix = np.zeros((len(queried_line), len(dictionary)))
    B_matrix = np.zeros((len(query_line), len(dictionary)))

    row = 0
    for document in corpus:
        for word_id, freq in document:
            A_matrix[row][word_id] = freq
        row = row + 1

    row = 0
    for document in corpus2:
        for word_id, freq in document:
            B_matrix[row][word_id] = freq
        row = row + 1

    sum_matrix = np.sum(np.vstack((A_matrix, B_matrix)), axis=0)
    probability_A = A_matrix / sum_matrix
    probability_B = B_matrix / sum_matrix

    sim = JS_Sim(probability_A, probability_B)

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim


def JS_Sim(A_set, B_set) -> pd.DataFrame:
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    # 开始计算JS相似度
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = JS_divergence(B_set[row], A_set[column])  # B_set为查询集，所以放前面
    return df


def JS_divergence(p, q):
    M = (p + q) / 2
    pk = np.asarray(p)
    pk2=np.asarray(q)
    a=0
    b=0
    if(np.sum(pk, axis=0, keepdims=True)!=0):
        a=0.5 * scipy.stats.entropy(p, M)
    if(np.sum(pk2, axis=0, keepdims=True) != 0):
        b= 0.5 * scipy.stats.entropy(q, M)

    return a+b  # 选用自然对数


# def evaluate_LDA(queried_file):
#     """
#     根据困惑度选取LDA的topic最佳个数
#     :param queried_file:语料文件路径
#     :return: 最佳LDA的topic number
#     """
#     queried_line = set_generation(queried_file)
#     # 被查询集生成词典和corpus
#     dictionary = corpora.Dictionary(queried_line)
#     corpus = [dictionary.doc2bow(text) for text in queried_line]
#     x = []  # x轴
#     perplexity_values = []  # 困惑度
#     for topic in range(10, 250, 10):
#         lda_model = models.LdaModel(corpus=corpus, num_topics=topic, id2word=dictionary)
#         x.append(topic)
#         perplexity_values.append(lda_model.log_perplexity(corpus))
#     plt.plot(x, perplexity_values, marker='o')
#     plt.xlabel('topic number')
#     plt.ylabel('perplexity values')
#     plt.show()
#     return x[perplexity_values.index(max(perplexity_values))]


def hellingerSim(A_set, B_set, topic_number):
    """
    计算两个集合中每条数据之间的Hellinger距离
    :param A_set: 被查询集
    :param B_set: 查询集
    :return: 一个 len(B_set) * len(A_set) 的 pandas.DataFrame
    """
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    A_matrix = np.zeros((len(A_set), topic_number))
    B_matrix = np.zeros((len(B_set), topic_number))

    # 将A_set和B_set分别转化为List[List[float]](e.i. 二维矩阵)
    row = 0
    for tu in A_set:
        for i in tu:
            A_matrix[row][i[0]] = i[1]
        row = row + 1
    row = 0
    for tu in B_set:
        for i in tu:
            B_matrix[row][i[0]] = i[1]
        row = row + 1

    # 开始计算Hellinger距离
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = HellingerDistance(B_matrix[row], A_matrix[column])  # B_matrix为查询集，所以放前面
    return df


def HellingerDistance(p, q):
    """
    计算HellingerDistance距离
    :param p:
    :param q:
    :return: float距离
    """
    return 1 - (1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q)))

def word2vec(queried_file,query_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)
    words_all = queried_line + query_line
    word2vecTrain(words_all)#训练词向量
    queried_line=np.asarray(queried_line)
    query_line = np.asarray(query_line)
    result=pd.DataFrame()
    flag=0
    for i in query_line:#代码行
        flag = flag + 1
        flag_j=0
        for j in queried_line:#查询集
            #print(wmd(i,j))
            result.loc[flag,flag_j]=wmd(i,j)#被查询集和查询集的相似度
            flag_j=flag_j+1
    return result

def word2vecTrain(text):
    model = Word2Vec()
    model.build_vocab(text)
    model.save('Word2VecModel.m')
    model.train(text, total_examples=model.corpus_count, epochs=model.epochs)
    model.wv.save_word2vec_format('Word2VecModel.vector', binary=False)


# 计算两个句子的wmd距离
def wmd(sent1, sent2):
    # sent1 = word_cut(sent1)
    # sent2 = word_cut(sent2)
    model = KeyedVectors.load_word2vec_format("Word2VecModel.vector")
    model.init_sims(replace=True)
    distance = model.wmdistance(sent1, sent2)
    return distance


def IR_based_feature_generation(fname, tname, output_fname=None):
    """
    生成两个制品之间链接向量(笛卡尔积个)
    :param fname: 制品1
    :param tname: 制品2
    :return: 链接向量，不带label
    """
    IR_based_feature = pd.DataFrame()
    options = [vsm_similarity, lsi_similarity, lda_similarity, bm25_score, JS_similarity]
    code_feature=['CN_MN_VN_CMT']
    flag = 0
    for option in options:
        sim = option(fname+code_feature[0]+'_clear.txt', tname)  # tname为查询集，fname为被查询集
        IR_based_feature[flag] = pd.concat([sim.iloc[i] for i in range(sim.shape[0])], axis=0,
                                ignore_index=True)  # 将sim所有行转化为IR_based_feature的一列
        print(option)
        flag = flag + 1

        sim = option(tname, fname+code_feature[0]+'_clear.txt')  # 查询集和被查询集交换
        IR_based_feature[flag] = pd.concat([sim.iloc[:, i] for i in range(sim.shape[1])], axis=0,
                                ignore_index=True)  # 将sim所有列转化为IR_based_feature的一列
        flag = flag + 1


    if output_fname is not None:
        IR_based_feature.to_excel(output_fname + ".xlsx")
    return IR_based_feature


if __name__ == '__main__':
    fname = "easyclinic_ID_UC/"
    tname = "easyclinic_ID_UC/uc_clear.txt"

    output_fname = "easyclinic_ID_UC/IR"
    IR_based_feature_generation(fname, tname, output_fname)
    # filepath="iTrust/iTrust_CAJP"
    # lists = os.walk(filepath)
    # for list in lists:
    #     root = list[0]
    #     files = list[2]
    #     for file in files:
    #         path = os.path.join(root, file)
    #         fname = path
    #         tname = "iTrust/UC_clear.txt"
    #         print(file[:-4])
    #         output_fname = "iTrust/IR_code_feature_CAJP/"+file[:-4]
    #         IR_based_feature_generation(fname, tname, output_fname)

