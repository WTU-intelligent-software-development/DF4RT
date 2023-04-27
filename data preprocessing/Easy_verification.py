# !/usr/bin/python
#  -*- coding:utf-8 -*-

from __future__ import division
import re
import codecs
import os
import time
import NLP_algorithm
#EasyClinic数据集比较真集
#Easy文件(UC_ID:total=26)(UC_uc:total=144)(ID_ID:total=59)(ID_TC:total=82)(ID_UC:total=26)(CC_CC:total=69)
#(CC_TC:total=204)(ID_CC:total=69)
def Easy_UCtoIDverification(select_num,total,truepath,fname,tname,ucname,ccname):

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

    #print('Selectivity:%f' % select_num)
    #print('precision:%f' % precision)
    #print('recall:%f' % recall)
    return select_num, precision, recall

def Easy_UCtoCCverification(select_num,total,truepath,fname,tname,ucname,ccname):

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

    #print('Selectivity:%f' % select_num)
    #print('precision:%f' % precision)
   # print('recall:%f' % recall)
    return select_num, precision, recall
if __name__ == '__main__':
    result = []
    num=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1]
    # Easy文件(UC_ID:total=26)(UC_uc:total=144)(ID_ID:total=59)(ID_TC:total=82)(ID_UC:total=26)(CC_CC:total=69)
    # (CC_TC:total=204)(ID_CC:total=69)
    truepath = '../sample-data/EasyClinic/ID_CC.txt'
    fname = '../sample-data/EasyClinic/CC_clear.txt'  # ID_FULL_clear.txt
    tname = '../sample-data/EasyClinic/ID_clear.txt'
    ucname = '../sample-data/EasyClinic/ID_TCName.txt'
    ccname = '../sample-data/EasyClinic/CC_TCName.txt'
    for i in num:
        select_num, re, pe=Easy_UCtoIDverification(i,69,truepath,fname,tname,ucname,ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_IDtoCC_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()
    result = []
    truepath = '../sample-data/EasyClinic/UC_ID.txt'
    fname = '../sample-data/EasyClinic/ID_clear.txt'
    tname = '../sample-data/EasyClinic/UC_clear.txt'
    ucname = '../sample-data/EasyClinic/UC_TCName.txt'
    ccname = '../sample-data/EasyClinic/ID_TCName.txt'
    for i in num:
        select_num, re, pe=Easy_UCtoIDverification(i,26,truepath,fname,tname,ucname,ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_UCtoID_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()
    result = []
    truepath = '../sample-data/EasyClinic/UC_UC.txt'
    fname = '../sample-data/EasyClinic/UC_clear.txt'
    tname = '../sample-data/EasyClinic/UC_clear.txt'
    ucname = '../sample-data/EasyClinic/UC_TCName.txt'
    ccname = '../sample-data/EasyClinic/UC_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoIDverification(i, 144, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_UCtoUC_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()

    result = []
    truepath = '../sample-data/EasyClinic/ID_UC.txt'
    fname = '../sample-data/EasyClinic/UC_clear.txt'
    tname = '../sample-data/EasyClinic/ID_clear.txt'
    ucname = '../sample-data/EasyClinic/ID_TCName.txt'
    ccname = '../sample-data/EasyClinic/UC_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoIDverification(i, 26, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_IDtoUC_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()

 # (ID_TC:total=82)(CC_CC:total=69)
    # (CC_TC:total=204)
    result = []
    truepath = '../sample-data/EasyClinic/ID_ID.txt'
    fname = '../sample-data/EasyClinic/ID_clear.txt'
    tname = '../sample-data/EasyClinic/ID_clear.txt'
    ucname = '../sample-data/EasyClinic/ID_TCName.txt'
    ccname = '../sample-data/EasyClinic/ID_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoIDverification(i, 59, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_IDtoID_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()

    # (ID_TC:total=82)
    result = []
    truepath = '../sample-data/EasyClinic/ID_TC.txt'
    fname = '../sample-data/EasyClinic/TC_clear.txt'
    tname = '../sample-data/EasyClinic/ID_clear.txt'
    ucname = '../sample-data/EasyClinic/ID_TCName.txt'
    ccname = '../sample-data/EasyClinic/TC_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoIDverification(i, 83, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_IDtoTC_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()

    # (CC_TC:total=204)
    result = []
    truepath = '../sample-data/EasyClinic/CC_TC.txt'
    fname = '../sample-data/EasyClinic/TC_clear.txt'
    tname = '../sample-data/EasyClinic/CC_clear.txt'
    ucname = '../sample-data/EasyClinic/CC_TCName.txt'
    ccname = '../sample-data/EasyClinic/TC_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoIDverification(i, 204, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_CCtoTC_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()

    # (CC_CC:total=69)
    result = []
    truepath = '../sample-data/EasyClinic/CC_CC.txt'
    fname = '../sample-data/EasyClinic/CC_clear.txt'
    tname = '../sample-data/EasyClinic/CC_clear.txt'
    ucname = '../sample-data/EasyClinic/CC_TCName.txt'
    ccname = '../sample-data/EasyClinic/CC_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoIDverification(i, 69, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_CCtoCC_vsm.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()

    # Easy文件(UC_TC:total=63)(TC_CC:total=204)(UC_CC:total=93)
    # Albergate total=53
    result = []
    truepath = '../sample-data/EasyClinic/UC_TC.txt'
    fname = '../sample-data/EasyClinic/TC_clear.txt'
    tname = '../sample-data/EasyClinic/UC_clear.txt'
    ucname = '../sample-data/EasyClinic/UC_TCName.txt'
    ccname = '../sample-data/EasyClinic/TC_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoCCverification(i, 63, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_UCtoTC_lsi.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()
    result = []
    truepath = '../sample-data/EasyClinic/TC_CC.txt'
    fname = '../sample-data/EasyClinic/CC_clear.txt'
    tname = '../sample-data/EasyClinic/TC_clear.txt'
    ucname = '../sample-data/EasyClinic/TC_TCName.txt'
    ccname = '../sample-data/EasyClinic/CC_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoCCverification(i, 204, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_TCtoCC_lsi.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()
    result = []
    truepath = '../sample-data/EasyClinic/UC_CC.txt'
    fname = '../sample-data/EasyClinic/CC_clear.txt'
    tname = '../sample-data/EasyClinic/UC_clear.txt'
    ucname = '../sample-data/EasyClinic/UC_TCName.txt'
    ccname = '../sample-data/EasyClinic/CC_TCName.txt'
    for i in num:
        select_num, re, pe = Easy_UCtoCCverification(i, 93, truepath, fname, tname, ucname, ccname)
        result.append((select_num, pe, re))
    outpath = '../sample-data/result/Easy_UCtoCC_lsi.txt'
    with open(outpath, 'w')as fp:
        fp.write('selsctivity' + ' ' + 'recall' + ' ' + 'Precision' + '\n')
        for list in result:
            for i in list:
                fp.write(str(i) + ' ')
            fp.write('\n')
    fp.close()