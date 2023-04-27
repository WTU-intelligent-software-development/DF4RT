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
import matplotlib.pyplot as plt
import itertools
from gensim.models.word2vec import Word2Vec
from gensim.similarities import WmdSimilarity
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
import warnings
import Specificity_Entropy
import Specificity_ICTF
import Specificity_IDF
import Specificity_QS
import Specificity_SCS
warnings.filterwarnings(action='ignore')

def QQ_feature_generation(fname, tname, output_fname=None):
    """
    生成两个制品之间链接向量(笛卡尔积个)
    :param fname: 制品1
    :param tname: 制品2
    :return: 链接向量，不带label
    """
    QQ_feature = pd.DataFrame()
    options = [Specificity_Entropy.Max_Entropy, Specificity_Entropy.Dev_Entropy, Specificity_Entropy.Med_Entropy,
    Specificity_Entropy.Avg_Entropy,Specificity_ICTF.AvgICDF,Specificity_ICTF.MaxICDF,Specificity_ICTF.DevICDF,
    Specificity_IDF.AvgIDF,Specificity_IDF.MaxIDF,Specificity_IDF.DevIDF,Specificity_SCS.KL_similarity,
    Specificity_QS.QS]
    code_feature=['CN_MN_VN_CMT']
    flag = 0
    for option in options:
        sim = option(fname+code_feature[0]+'_clear.txt', tname)  # tname为查询集，fname为被查询集
        QQ_feature[flag] = pd.concat([sim.iloc[i] for i in range(sim.shape[0])], axis=0,
                                ignore_index=True)  # 将sim所有行转化为IR_based_feature的一列
        print(option)
        flag = flag + 1

        sim = option(tname, fname+code_feature[0]+'_clear.txt')  # 查询集和被查询集交换
        QQ_feature[flag] = pd.concat([sim.iloc[:, i] for i in range(sim.shape[1])], axis=0,
                                ignore_index=True)  # 将sim所有列转化为IR_based_feature的一列
        flag = flag + 1


    if output_fname is not None:
        QQ_feature.to_excel(output_fname + ".xlsx")
    return QQ_feature



if __name__ == '__main__':
    fname = "../easyclinic_ID_UC/"
    tname = "../easyclinic_ID_UC/uc_clear.txt"
    output_fname = "../easyclinic_ID_UC/QQ"
    QQ_feature_generation(fname, tname, output_fname)
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

