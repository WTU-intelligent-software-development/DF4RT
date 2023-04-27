# -*-coding:utf-8-*-

from gensim import corpora, similarities
import codecs
from gensim import models
import time
import re
# VSM相似度计算
def vsm_similarity(fname,tname,select_num):
    #生成训练集
    f = codecs.open(fname, 'rb', 'utf-8')
    lines = f.readlines()
    trainline = []
    for line in lines:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        trainline.append(word)
    testline = []
    #生成测试集
    ft=codecs.open(tname, 'rb', 'utf-8')
    lines_T = ft.readlines()
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        testline.append(word)


    # 训练集生成词典和corpus
    dictionary = corpora.Dictionary(trainline)#给每个词分配一个独一无二的编号
    corpus = [dictionary.doc2bow(text) for text in trainline]#在每句话中每个词语出现的频率
    #print(corpus)

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 相似度（根据语料库）
    corpus_sim = similarities.MatrixSimilarity(corpus_tfidf)

    #得到源文档对目标文档的相似度
    sim = []
    for i in range(len(testline)):
        test_bow = dictionary.doc2bow(testline[i])#对不同单词进行计数并返回其编号
        test_tfidf = tfidf_model[test_bow]#算每个文档的加权值
        test_sim = corpus_sim[test_tfidf]
        sim.append(test_sim)

    #按照文本相似度进行排序
    result=[]
    for i in range(len(sim)):
        for j in range(len(sim[i])):
                    result.append((i+1,j+1,sim[i][j]))
    result=sorted(result, key=(lambda x: x[2]), reverse=True)

    #选取候选链接
    ch_result=[]
    num=1
    total=len(result)*select_num
    for i in range(len(result)):
        if num<total or num ==total:
            ch_result.append(result[i])
            num=num+1
        else:
            break
   # print(len(result))
   # print(total)
   # print('vsm文本相似度(FRS,SRS)')
    print(ch_result)
    return ch_result


# LSI相似度计算
def lsi_similarity(fname,tname,select_num):
    # 读取文本文件
    #fname = '../../sample-data/' + filename
    f = codecs.open(fname, 'rb', 'utf-8')
    lines = f.readlines()
    trainline = []
    for line in lines:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        trainline.append(word)

        # 生成测试集
        testline = []
        ft = codecs.open(tname, 'rb', 'utf-8')
        lines_T = ft.readlines()
        for line in lines_T:
            word = line.split(' ')
            word = [re.sub('\s', '', i) for i in word]
            testline.append(word)
    #trainline = lsi_lines[:len_train]
    #testline = lsi_lines[len_train:]

    # 生成词典和corpus
    dictionary = corpora.Dictionary(trainline)
    corpus = [dictionary.doc2bow(text) for text in trainline]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 生成lsi主题
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary)
    corpus_lsi = lsi_model[corpus_tfidf]

    # 相似度
    corpus_sim = similarities.MatrixSimilarity(corpus_lsi)

    sim = []
    for i in range(len(testline)):
        test_bow = dictionary.doc2bow(testline[i])
        test_tfidf = tfidf_model[test_bow]
        test_lsi = lsi_model[test_tfidf]
        test_sim = corpus_sim[test_lsi]
        #print(test_sim)
        sim.append(test_sim)

        # 按照文本相似度进行排序
        result = []
        for i in range(len(sim)):
            for j in range(len(sim[i])):
                result.append((i + 1, j + 1, sim[i][j]))
        result = sorted(result, key=(lambda x: x[2]), reverse=True)

        # 选取候选链接
        ch_result = []
        num = 1
        total = len(result) * select_num
        for i in range(len(result)):
            if num < total or num == total:
                ch_result.append(result[i])
                num = num + 1
            else:
                break
    #print(result)
    #print(total)
   # print('Lsi文本相似度')
    #print(ch_result)
    return ch_result


if __name__ == '__main__':

    #fname = '../sample-data/eTOUR/UC_clear.txt'
    #tname = '../sample-data/eTOUR/MN_CMT_clear.txt'

    fname = '../sample-data/EasyClinic/UC_clear.txt'
    tname = '../sample-data/EasyClinic/ID_clear.txt'

   # fname = '../sample-data/eTOUR/MN_CMT_clear.txt'
    #tname = '../sample-data/eTOUR/UC_clear_tea.txt'

    #fname = '../sample-data/EBT/tc_clear.txt'
    #tname = '../sample-data/EBT/re_clear.txt'

    #fname = '../sample-data/iTrust/MN_CMT_clear.txt'
    #tname = '../sample-data/iTrust/UC_clear.txt'
    sim=vsm_similarity(fname,tname,0.6)

    print(sim)