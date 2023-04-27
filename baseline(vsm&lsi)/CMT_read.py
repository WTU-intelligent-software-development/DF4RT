#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import codecs
import re
import nltk
from nltk.corpus import stopwords
import xlrd
import pandas as pd
# eTOUR代码文件读取（注释）
def readcode_1(filepath, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fname = savepath + 'CMT_num.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')

    cfname = savepath + 'CMT_Name.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')

    #lists = os.listdir(filepath)  # 列出文件夹下所有的目录与文件
    lists = os.walk(filepath)
    result = 0
    for list in lists:
        print (list)
        root = list[0]
        files = list[2]
        for file in files:
            path = os.path.join(root, file)
            #print (path)

    #读取需要的文件
            if os.path.isfile(path):
                f = codecs.open(path, 'rb',encoding='ISO-8859-1')
                lines = f.readlines()
                comment = []
                flag = 0
                flag_1=0
                index = 0
                num = 0
                code_num = 0

    #eTPUR提取注释
                for line in lines:
                    #块注释
                    if line != '':
                        if line.strip().strip(')') != '' and line.strip().strip('(') != '' and line.strip().strip(
                                '/ **') != '' and line.strip().strip('*') != '' and line.strip().strip(
                                '/ *') != '' and line.strip().strip('* /') != '':
                            code_num = code_num + 1
                    if '/ **' in line:
                        flag=1
                        if re.sub('/ \*\*', '', line).strip()!='':
                            comment.append(re.sub('/ \*\*', '', line).strip()+' ')
                            num = num + 1
                    if flag==1 and '*' in line and '/ **' not in line:
                        if re.sub('\*', '', line).strip()!='':
                            comment.append(re.sub('\*', '', line).strip()+' ')
                            num = num + 1
                    if '* /' in line:
                        if re.sub('/* /', '', line).strip()!='':
                            comment.append(re.sub('/* /', '', line).strip()+' ')
                            num = num + 1
                        flag=0
                    if '/ *' in line and '/ **' not in line:
                        flag_1=1
                        if re.sub('/ *', '', line).strip()!='':
                            comment.append(re.sub('/ *', '', line).strip()+' ')
                            num = num + 1

                    if flag_1 == 1 and '*' in line and '* /' not in line:
                        if re.sub('\*', '', line).strip()!='':
                            comment.append(re.sub('\*', '', line).strip()+' ')
                            num = num + 1
                    if '* /' in line:
                        if re.sub('/* /', '', line).strip() != '':
                            comment.append(re.sub('/* /', '', line).strip() + ' ')
                            num = num + 1
                        flag_1 = 0
                    if '/ / 'in line: # 行注释
                        comment.append(re.sub('/ / ', '', line).strip()+' ')
                        num = num + 1
                #itrust注释
                '''all=0
                for line in lines:
                    #块注释
                    all=all+1
                    if line.strip()!='' :
                        if line.strip().strip('{') != '' and line.strip().strip('}') != '' and line.strip().strip('/**') != '' and line.strip().strip('*') != '' and line.strip().strip('*/') != '' and line.strip().strip('/*') != '':
                            code_num = code_num+1
                        #print(code_num)
                    if '/*' in line and '/**' not in line:
                        flag=1
                        if re.sub('/*', '', line).strip() != '':
                            comment.append(re.sub('/*', '', line).strip())
                            num=num+1
                    if flag == 1 and '*' in line and '/*' not in line:
                        if re.sub('\*', '', line).strip() != '' and '*/' not in line:
                            comment.append(re.sub('\*', '', line).strip())
                            num = num + 1
                        if '*/' in line:
                            flag = 0
                            if re.sub('\*/', '', line).strip() != '':
                                comment.append(re.sub('\*/', '', line).strip())
                                num = num + 1

                    if '/**' in line:
                        flag_1=1
                        if re.sub('/\*\*', '', line).strip()!='':
                            comment.append(re.sub('/*\\*', '', line).strip())
                            num = num + 1

                    if flag_1==1 and '*' in line and '/**' not in line:
                        if re.sub('\*', '', line).strip()!='' and '*/'not in line:
                            comment.append(re.sub('\*', '', line).strip())
                            num = num + 1

                        if '*/' in line:
                            flag_1 = 0
                            if re.sub('\*/', '', line).strip()!='':
                                comment.append(re.sub('\*/', '', line).strip())
                                num = num + 1
                    if '//'in line:
                        comment.append(re.sub('//', '', line).strip())
                        num =num+1'''
                '''for line in lines:
                    if line.strip() != '' :
                        if line.strip().strip('{') != '' and line.strip().strip('}') != '' and line.strip().strip(
                                '/*') != '' and line.strip().strip('*') != '' and line.strip().strip(
                                '*/') != '' '':
                            code_num = code_num + 1
                    if '/*' in line:
                        flag = 1
                        if re.sub('/\*', '', line).strip() != '':
                            comment.append(re.sub('/\*', '', line).strip())
                            num = num + 1
                        else:
                            code_num = code_num - 1
                    if flag == 1 and '*' in line and '/**' not in line:
                        if re.sub('\*', '', line).strip() != '' and '*/' not in line:
                            comment.append(re.sub('\*', '', line).strip())
                            num = num + 1
                        else:
                            code_num = code_num - 1
                        if '*/' in line:
                            if re.sub('\*/', '', line).strip() != '':
                                comment.append(re.sub('\*/', '', line).strip())
                                num = num + 1
                            else:
                                code_num = code_num - 1
                            flag = 0
                    if '//' in line:  # 行注释
                        line=line.split('//')
                        comment.append(re.sub('//', '', line[1]).strip())
                        num = num + 1'''
                #print(comment)
                print(all)
                print(code_num)
                print(num)
                result=result+num / code_num
                print(num / code_num )
                if comment != '':
                    fc.write(file + '\n')
                    for line in comment:
                        fp.write(line+' ')
                    fp.write('\n')


    print(num / (code_num + num))
    print(result/116)
if __name__ == '__main__':

    '''filepath = '../CoESTData/iTrust/iTrust-code/'
    savepath = '../sample-Data/iTrust/'''

    '''filepath = '../CoESTData/SMOS/cc/'
    savepath = '../sample-Data/SMOS/'''
    filepath = '../CoESTData/eTOUR/CC/'
    savepath = '../sample-Data/eTOUR/'
    '''filepath = '../CoESTData/Albergate/to_be_traced_source_code/'
    savepath = '../sample-Data/Albergate/'''

    readcode_1(filepath, savepath)