#!/usr/bin/python
# -*- coding:utf-8 -*-
#读取easyclinic的代码文件
import os
import codecs
import re
import nltk
from nltk.corpus import stopwords
import xlrd
import pandas as pd
#读取代码的特征值
def readcode_cmt(filepath, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    '''fname = savepath + 'CMT2.txt'
    fp1 = codecs.open(fname, 'a+', 'utf-8')
    fname = savepath + 'MN.txt'
    fp2 = codecs.open(fname, 'a+', 'utf-8')
    fname = savepath + 'VN.txt'
    fp3 = codecs.open(fname, 'a+', 'utf-8)
    '''
    fname = savepath + 'easyclinic.txt'
    fp= codecs.open(fname, 'a+', 'utf-8')
    cfname = savepath + 'easyclinic_Name.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')
    lists = os.walk(filepath)
    result=0
    for list in lists:
        print(list)
        root = list[0]
        files = list[2]
        for file in files:
            path = os.path.join(root, file)
            # print (path)
            code_num=0
            num=0
            # 读取需要的文件
            if os.path.isfile(path):
                f = codecs.open(path, 'rb', encoding='ISO-8859-1')
                lines = f.readlines()
                for line in lines:
                    index = 0
                    '''if line.strip()!=' ':
                        code_num = code_num + 1'''
                    if 'Private' in line or('(' not in line and 'Class' not in line and 'Version' not in line and'Attributes' not in line and 'Name Access Description' not in line and 'Methods' not in line and ' 0 01 000' not in line) :
                        num=num+1
                        index = 1
                        #fp1.write(line.strip('\r\n').replace('Description', ''))
                        #fp1.write(' ')
                        if 'Private' in line:
                            fp.write('Private' + line.split(' ')[1])
                            fp.write(' ')
                    if index!=1:
                        fp.write(line.strip('\r\n'))
                        fp.write(' ')
                    '''if 'Private' in line:
                        fp2.write('Private'+line.split(' ')[1])
                        fp2.write(' ')
                    if '(' in line:
                        fp3.write(line.split('(')[0].replace('Signature',''))
                        fp3.write(' ')'''

                '''result=result+num/code_num
                print(num)
                print(code_num)'''
            fp.write('\r\n')
    '''result=result/47
    print(result)'''


            #fp2.write('\r\n')
            #fp1.write('\n')
            #fp3.write('\r\n')

    fp.close()
    #fp1.close()
    #fp2.close()
    #fp3.close()
if __name__ == '__main__':
    filepath='../CoESTData/EasyClinic/EasyClinic/EasyClinicDataset/2 - docs (English)/4 - class description/'
    savepath='../sample-data/XR/'
    readcode_cmt(filepath, savepath)