# !/usr/bin/python 
#  -*- coding:utf-8 -*-

from __future__ import division
import re
import codecs
import os
import time
import NLP_algorithm


def WARC_FRStoSRSverification(select_num):
    truepath = '../sample-data/WARC/FRStoSRS.txt'
    fname = '../sample-data/WARC/SRS_clear.txt'
    tname = '../sample-data/WARC/FRS_clear.txt'
    result=NLP_algorithm.vsm_similarity(fname,tname,select_num)
    #result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    f = codecs.open(truepath, 'rb')
    truelines = f.readlines()

    FRS = []
    SRS = [[1, 2, 3, 4, 5, 7], [4, 5, 6, 11, 21, 68], [7], [8, 9, 10], [11, 14], [11, 12, 14], [15], [16], [16, 18], [17], [19], [19], [20], [21, 22, 23, 24, 25, 26], [25], [21, 22, 23, 24, 25, 26], [27, 28], [29, 30], [], [31, 32], [34, 36], [37], [38], [39, 40], [41, 42, 43], [44], [45, 46], [47], [48], [49], [51], [52], [53], [54], [55, 56], [56], [57], [59], [51, 60], [61], [62], [63, 64]]
    total = 78
    p = 0
    r = 0
    num=1
    for trueline in truelines:
        if '%' not in str(trueline):
            s = []
            words = trueline.decode().split('\t')
            for word in words:
                if 'FR' in word:
                    #FRS.append(int(re.sub('\D', '', word)))
                    FRS.append(num)
                    num=num+1
                else:
                    w = word.split(' ')
                    '''for i in w:
                        if 'SRS' in i:
                            s.append(int(re.sub('\D', '', i)))'''
            SRS.append(s)
    print(SRS)

    dict = {}
    t_list = []
    for j in range(len(result)):
        # 判断是否存在于该SRS所对应的SRS序列中
        #print(result[i][1])
        #print(SRS[result[i][0]-1])
        if result[j][1] in SRS[result[j][0]-1]:
            #print('%s----->%s' % ( result[i][0],result[i][1]))
            p += 1
            t_list.append((result[j][0], result[j][1]))
            # 计算召回率
            if (result[j][0], result[j][1]-1) not in dict.keys():
                dict[result[j][0], result[j][1]-1] = 1
                r += 1
    #print(p)
    #print(len(result))
    precision = p / len(result)
    recall = r / total
    print('Selectivity:%f'%select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall

def WARC_NFRtoSRSverification(select_num):
    truepath = '../sample-data/WARC/NFRtoSRS.txt'
    fname = '../sample-data/WARC/SRS_clear.txt'
    tname = '../sample-data/WARC/NFR_clear.txt'

    f = codecs.open(truepath, 'rb')
    truelines = f.readlines()
    result = NLP_algorithm.vsm_similarity(fname, tname, select_num)
   # result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    NFR = []
    SRS = [[49], [67], [69, 70, 71, 72, 80, 84, 85, 89], [31, 41, 45, 47, 48, 49, 52, 73, 74], [75], [70, 71, 72], [], [76], [77], [70, 78], [79, 80], [51, 59, 60, 63], [], [39, 40], [82, 83, 84], [84, 85], [80, 88, 89], [59], [51, 59, 60], [61, 62, 63, 64], [51, 59, 60, 61, 62, 63, 64]]
    total = 58
    p = 0
    r = 0
    num=1
    for trueline in truelines:
        if '%' not in str(trueline):
            s = []
            words = trueline.decode().split('\t')
            print(words)
            for word in words:
                if 'NFR' in word:
                    NFR.append(num)
                    num=num+1
                    #NFR.append(int(re.sub('\D', '', word)))
                else:
                    w = word.split(' ')
                    '''for i in w:
                        if 'SRS' in i:
                            s.append(int(re.sub('\D', '', i)))'''
            SRS.append(s)
    print(NFR)
    print(SRS)
    dict = {}
    t_list = []
    for j in range(len(result)):
        # 判断是否存在于该SRS所对应的SRS序列中
        if result[j][1] in SRS[result[j][0] - 1]:
            # print('%s----->%s' % ( result[i][0],result[i][1]))
            p += 1
            t_list.append((result[j][0], result[j][1]))
            # 计算召回率
            if (result[j][0], result[j][1] - 1) not in dict.keys():
                dict[result[j][0], result[j][1] - 1] = 1
                r += 1

    precision = p / len(result)
    recall = r / total

    print('Selectivity:%f' % select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall



#Easy文件(UC_TC:total=63)(TC_CC:total=204)(UC_CC:total=93)
#Albergate total=53
def Easy_UCtoCCverification(select_num,total):
    truepath = '../sample-data/EasyClinic/UC_CC.txt'
    fname = '../sample-data/EasyClinic/CC_clear.txt'
    tname = '../sample-data/EasyClinic/UC_clear.txt'
    # 处理真集
    f = codecs.open(truepath, 'rb', 'utf-8')
    truelines = f.readlines()
    p = 0
    r = 0
    true_class = []
    for trueline in truelines:
        words = trueline.split(' ')
        for i in range(len(words)):
            if i>0 and '\r\n' not in words[i]:
                true_class.append((words[0],words[i]))
    #print('true')
    print(len(true_class))
    # 文件名
    ucname = '../sample-data/EasyClinic/UC_TCName.txt'
    ccname = '../sample-data/EasyClinic/CC_TCName.txt'


    result = NLP_algorithm.vsm_similarity(fname, tname, select_num)
   # result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    uc = []
    cc = []
    f = codecs.open(ucname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        uc.append(trueline.replace('\r\n', '',))

    f = codecs.open(ccname, 'rb', 'utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        cc.append(trueline.replace('\r\n', '',))

    result_uToc = []
    for i in range(len(result)):
        result_uToc.append((uc[result[i][0] - 1].lower(), cc[result[i][1] - 1].lower()))

    # 查找有多少个真集数
    for i in range(len(result_uToc)):
        for j in range(len(true_class)):
            if result_uToc[i][0] == true_class[j][0] and result_uToc[i][1] == true_class[j][1]:
                print(result_uToc[i][0], result_uToc[i][1])
                p = p + 1

    precision = p / len(result)
    recall = p / total

    print('Selectivity:%f' % select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall

#Easy文件(UC_ID:total=26)(UC_uc:total=144)(ID_ID:total=59)(ID_TC:total=82)(ID_UC:total=26)(CC_CC:total=69)
#(CC_TC:total=204)(ID_CC:total=69)
def Easy_UCtoIDverification(select_num,total):
    truepath = '../sample-data/EasyClinic/ID_CC.txt'
    fname = '../sample-data/EasyClinic/CC_clear.txt'#ID_FULL_clear.txt
    tname = '../sample-data/EasyClinic/ID_clear.txt'
    # 处理真集
    f = codecs.open(truepath, 'rb', 'utf-8')
    truelines = f.readlines()
    p = 0
    true_class = []
    for trueline in truelines:
        words = trueline.split(':')
        if words[1].strip('\r\n')!='' :
            words2=words[1].split(' ')
            for j in range(len(words2)):
                if words2[j].strip('\r\n')!='':
                    true_class.append((words[0],words2[j]))

    print(true_class)
    print(len(true_class))
    # 文件名
    ucname = '../sample-data/EasyClinic/ID_TCName.txt'
    ccname = '../sample-data/EasyClinic/CC_TCName.txt'


    result = NLP_algorithm.vsm_similarity(fname, tname, select_num)
    #result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    uc = []
    cc = []
    f = codecs.open(ucname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        uc.append(trueline.replace('\r\n', '',))

    f = codecs.open(ccname, 'rb', 'utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        cc.append(trueline.replace('\r\n', '',))
    print(uc)
    print(cc)
    result_uToc = []
    for i in range(len(result)):
        result_uToc.append((uc[result[i][0] - 1].lower(), cc[result[i][1] - 1].lower()))


    # 查找有多少个真集数
    for i in range(len(result_uToc)):
        for j in range(len(true_class)):
            if result_uToc[i][0] == true_class[j][0] and result_uToc[i][1] == true_class[j][1]:
                print(result_uToc[i][0], result_uToc[i][1])
                p = p + 1

    precision = p / len(result)
    recall = p / total

    print('Selectivity:%f' % select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall


#eTOUR文件
def eTOUR_UCtoCCverification(select_num ):
    truepath = '../sample-data/eTOUR/AnswerSet.txt'
    fname = '../sample-data/eTOUR/MN_CMT_clear.txt'
    tname = '../sample-data/eTOUR/UC_FULL_clear.txt'

    #处理真集
    f = codecs.open(truepath, 'rb','utf-8')
    truelines = f.readlines()
    total = 308
    p = 0

    true_class = []
    for trueline in truelines:
        words = trueline.split('\t')
        true_class.append((words[0].lower(), words[1].lower().replace('\r\n', '')))
    #print(len(true_class))

    #文件名
    ucname = '../sample-data/eTOUR/UC_TName.txt'
    ccname =  '../sample-data/eTOUR/CC_TName.txt'

    result = NLP_algorithm.vsm_similarity(fname, tname, select_num)
    #result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    print(result)
    uc=[]
    cc = []
    f = codecs.open(ucname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        uc.append(trueline.replace('\r\n',''))

    f = codecs.open(ccname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        cc.append(trueline.replace('\r\n',''))

    result_uToc=[]
    for i in range(len(result)):
        result_uToc.append((uc[result[i][0]-1].lower(),cc[result[i][1]-1].lower()))
    print(result_uToc)
   # print(result_uToc[0][1])
    #print(true_class[0][0])

    #print(true_class[0][1])
    #查找有哪些没找到
    not_find=[]
    #查找有多少个真集数
    for i in range(len(result_uToc)):
        for j in range(len(true_class)):
            if result_uToc[i][0].lower()==true_class[j][0].lower() and result_uToc[i][1].lower()==true_class[j][1].lower():
                p = p + 1
                if result_uToc[i][0].lower()=='uc20.txt' or result_uToc[i][0].lower()=='uc31.txt' or result_uToc[i][0].lower()=='uc32.txt' or result_uToc[i][0].lower()=='uc54.txt':
                    print(result_uToc[i][0],result_uToc[i][1])
                #not_find.append((result_uToc[i][0].split('.'))[0])

    not_find.sort(key=lambda x:int(x[2:]))
    precision = p / len(result)
    recall = p/ total
    print(not_find)
    print('Selectivity:%f' % select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall

#EBT文件
def EBT_REtoTCverification(select_num):
    truepath = '../sample-data/EBT/AnswersRequirements2Testcases.txt'
    fname = '../sample-data/EBT/tc_clear.txt'
    tname = '../sample-data/EBT/re_clear.txt'
    #处理真集
    f = codecs.open(truepath, 'rb','utf-8')
    truelines = f.readlines()
    total = 51
    p = 0
    true_class = []
    for trueline in truelines:
        words = trueline.split(' ')
        print(words)
        for i in range(len(words)):
            if i > 0 and words[i].strip('\r\n')!=' ':
                true_class.append((words[0], words[i].strip('\r\n')))
    print(true_class)

    #文件名
    ucname = '../sample-data/EBT/re_TName.txt'
    ccname =  '../sample-data/EBT/tc_TName.txt'


    result = NLP_algorithm.vsm_similarity(fname, tname, select_num)
    #result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    uc=[]
    cc = []
    f = codecs.open(ucname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        uc.append(trueline.replace('\r\n',''))

    f = codecs.open(ccname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        cc.append(trueline.replace('\r\n',''))

    result_uToc=[]
    for i in range(len(result)):
        result_uToc.append((uc[result[i][0]-1].lower(),cc[result[i][1]-1].lower()))
   # print(result_uToc[0][1])
    #print(true_class[0][0])

    #print(true_class[0][1])


    #查找有多少个真集数
    for i in range(len(result_uToc)):
        for j in range(len(true_class)):
            if result_uToc[i][0].lower()==true_class[j][0].lower() and result_uToc[i][1].lower()==true_class[j][1].lower():
                p = p + 1
                print(result_uToc[i][0],result_uToc[i][1])

    precision = p / len(result)
    recall = p/ total

    print('Selectivity:%f' % select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall

#iTrust
def iTrust_UCtoCCverification(select_num):
    truepath = '../sample-data/iTrust/answer_req_javacode.txt'
    fname = '../sample-data/iTrust/MN_CMT_clear.txt'
    tname = '../sample-data/iTrust/UC_clear.txt'

    #处理真集
    f = codecs.open(truepath, 'rb','utf-8')
    truelines = f.readlines()
    total = 418
    p = 0

    true_class = []
    for trueline in truelines:
        words = trueline.split(' ')#'\t'
        true_class.append((words[0].lower(), words[1].lower().replace('\r\n', '')))
    print(len(true_class))

    #文件名
    ucname = '../sample-data/iTrust/UC_TCName.txt'
    ccname =  '../sample-data/iTrust/CMT_Name.txt'


   # result = NLP_algorithm.vsm_similarity(fname, tname, select_num)
    result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    print(result)
    uc=[]
    cc = []
    f = codecs.open(ucname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        uc.append(trueline.replace('\r\n','').split('.')[0])

    f = codecs.open(ccname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        cc.append(trueline.replace('\n','').split('.')[0])

    result_uToc=[]
    for i in range(len(result)):
        result_uToc.append((uc[result[i][0]-1].lower(),cc[result[i][1]-1].lower()))
    print(result_uToc)
   # print(result_uToc[0][1])
    #print(true_class[0][0])

    #print(true_class[0][1])
    #查找有哪些没找到
    not_find=[]
    #查找有多少个真集数
    for i in range(len(result_uToc)):
        for j in range(len(true_class)):
            if result_uToc[i][0].lower()==true_class[j][0].lower() and result_uToc[i][1].lower()==true_class[j][1].lower():
                p = p + 1
                if result_uToc[i][0].lower()=='uc20.txt' or result_uToc[i][0].lower()=='uc31.txt' or result_uToc[i][0].lower()=='uc32.txt' or result_uToc[i][0].lower()=='uc54.txt':
                    print(result_uToc[i][0],result_uToc[i][1])
                #not_find.append((result_uToc[i][0].split('.'))[0])

    not_find.sort(key=lambda x:int(x[2:]))
    precision = p / len(result)
    recall = p/ total
    print(not_find)
    print('Selectivity:%f' % select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall



#   CM1
def CM1_UCtoCCverification(select_num):

    truepath = '../sample-data/CM1/CM1-answerSet.txt'
    fname = '../sample-data/CM1/targetArtifacts_clear.txt'
    tname = '../sample-data/CM1/sourceArtifacts_clear.txt'

    #处理真集
    f = codecs.open(truepath, 'rb','utf-8')
    truelines = f.readlines()
    total = 45
    p = 0

    true_class = []
    for trueline in truelines:
        words = trueline.split(' ')
        true_class.append((words[0].lower(), words[1].lower().replace('\n', '')))
    #print(len(true_class))

    #文件名
    ucname = '../sample-data/CM1/sourceArtifacts_name.txt'
    ccname =  '../sample-data/CM1/targetArtifacts_name.txt'


    result = NLP_algorithm.vsm_similarity(fname, tname, select_num)
    #result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    print(result)
    uc=[]
    cc = []
    f = codecs.open(ucname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        uc.append(trueline.replace('\n',''))

    f = codecs.open(ccname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        cc.append(trueline.replace('\n',''))

    result_uToc=[]
    for i in range(len(result)):
        result_uToc.append((uc[result[i][0]-1].lower(),cc[result[i][1]-1].lower()))
    print(result_uToc)
   # print(result_uToc[0][1])
    #print(true_class[0][0])

    #print(true_class[0][1])
    #查找有哪些没找到
    not_find=[]
    #查找有多少个真集数
    for i in range(len(result_uToc)):
        for j in range(len(true_class)):
            if result_uToc[i][0].lower()==true_class[j][0].lower() and result_uToc[i][1].lower()==true_class[j][1].lower():
                p = p + 1
                if result_uToc[i][0].lower()=='uc20.txt' or result_uToc[i][0].lower()=='uc31.txt' or result_uToc[i][0].lower()=='uc32.txt' or result_uToc[i][0].lower()=='uc54.txt':
                    print(result_uToc[i][0],result_uToc[i][1])
                #not_find.append((result_uToc[i][0].split('.'))[0])

    not_find.sort(key=lambda x:int(x[2:]))
    precision = p / len(result)
    recall = p/ total
    print(not_find)
    print('Selectivity:%f' % select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall


#Albergate total=53
#GANNT total = 68
def Albergate_verification(select_num):
    truepath = '../sample-data/GANNT/AnswerSetHighToLow.txt'
    fname = '../sample-data/GANNT/low_Full_clear.txt'
    tname = '../sample-data/GANNT/high_Full_clear.txt'
    '''truepath = '../sample-data/Albergate/AnswerMatrix.txt'
    fname = '../sample-data/Albergate/MN_CMT_clear.txt'
    tname = '../sample-data/Albergate/re_Full_clear.txt'''''
    # 处理真集
    f = codecs.open(truepath, 'rb', 'utf-8')
    truelines = f.readlines()
    #total = 53
    total = 68
    p = 0
    true_class = []
    for trueline in truelines:
        words = trueline.split(' ')
        true_class.append((words[0].lower(),words[1].replace('\r\n','').lower()))
    #print('true')
    print(len(true_class))
    print(true_class)
    # 文件名
    ucname = '../sample-data/GANNT/high_TCName.txt'
    ccname = '../sample-data/GANNT/low_TCName.txt'
    '''ucname = '../sample-data/Albergate/re_TCName.txt'
    ccname = '../sample-data/Albergate/CMT_Name.txt'''''

    #result = NLP_algorithm.vsm_similarity(fname, tname, select_num)
    result = NLP_algorithm.lsi_similarity(fname, tname, select_num)
    uc = []
    cc = []
    f = codecs.open(ucname, 'rb','utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        uc.append(trueline.replace('\r\n', '',))
    f = codecs.open(ccname, 'rb', 'utf-8')
    truelines = f.readlines()
    for trueline in truelines:
        cc.append(trueline.replace('\r\n', '',))
        #cc.append(trueline.replace('\n', '', ))

    result_uToc = []
    for i in range(len(result)):
        result_uToc.append((uc[result[i][0] - 1].lower(), cc[result[i][1] - 1].lower()))
    print(result_uToc)
    # 查找有多少个真集数
    for i in range(len(result_uToc)):
        for j in range(len(true_class)):
            if result_uToc[i][0] == true_class[j][0] and result_uToc[i][1] == true_class[j][1]:
                print(result_uToc[i][0], result_uToc[i][1])
                p = p + 1

    precision = p / len(result)
    recall = p / total

    print('Selectivity:%f' % select_num)
    print('precision:%f' % precision)
    print('recall:%f' % recall)
    return select_num, precision, recall


if __name__ == '__main__':
    result = []
    num=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1]

    '''for i in num:
        select_num, re, pe=WARC_FRStoSRSverification(i)
        result.append((select_num, pe, re))

    outpath = '../sample-data/result/WARC_vsm(FRStoSRS).txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()
    result = []
    for i in num:
        select_num, re, pe = WARC_NFRtoSRSverification(i)
        result.append((select_num, pe, re))

    outpath = '../sample-data/result/WARC_vsm(NFRtoSRS).txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()


    result = []
    for i in num:
        select_num, re, pe = eTOUR_UCtoCCverification(i)
        result.append((select_num, pe, re))

    outpath = '../sample-data/result/eTOUR_vsm.txt'#取了全部的uc vsm2/只取了uc的第二行
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()
    result = []
    for i in num:
        select_num, re, pe =  EBT_REtoTCverification(i)
        result.append((select_num, pe, re))

    outpath = '../sample-data/result/EBT_REtoTC_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()'''
    result = []
    for i in num:
        select_num, re, pe = iTrust_UCtoCCverification(i)
        result.append((select_num, pe, re))

    outpath = '../sample-data/result/iTrust_lsi.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()
    result = []
    '''for i in num:
        select_num, re, pe= CM1_UCtoCCverification(i)
        result.append((select_num, pe, re))

    outpath = '../sample-data/result/CM1_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()'''
    '''result = []
    for i in num:
        select_num, re, pe= Albergate_verification(i)
        result.append((select_num, pe, re))

    outpath = '../sample-data/result/GANNT_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()
    result = []
    for i in num:
        select_num, re, pe = Albergate_verification(i)
        result.append((select_num, pe, re))'''






