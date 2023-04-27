#!/usr/bin/python 
# -*- coding:utf-8 -*-
import os
import codecs
import re

import pandas as pd
# WARC文件读取
def CodeTranslate(filepath, savepath):
    # file_dir = 'E:/ProjectTest/new/IceBreaker/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fname = savepath + 'SRS_Full.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')
#存储表名
    cfname = savepath + 'SRS_TCName.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')

    lists = os.listdir(filepath)  # 列出文件夹下所有的目录与文件
    for list in lists:
        if 'txt' in list:
            fc.write(list + '\r\n')  # 将文件名保存至TCName文本文件中
            path = os.path.join(filepath, list)
            print (path)#输出每个文件的地址

            if os.path.isfile(path):#如果该地址存在，读取文档中的内容
                f = codecs.open(path, 'rb')
                lines = f.readlines()


                # 处理GANNT数据集时
                '''if 'high' in filepath:
                    words = line.split(';', 1)
                else:
                    words = line.split('.', 1)
                str = words[0]'''
                # 将需求或软件制品文本保存至文本文件
                for line in lines:
                    words = line.decode().split('- ')
                    line = words[1]
                    st = re.sub('\r\n', ' ', str(line))
                    print(st)
                    fp.write(st)
                fp.write('\r\n')

#EasyClinic文件读取
def EasyClinicTranslate(filepath, savepath):
    # file_dir = 'E:/ProjectTest/new/IceBreaker/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fname = savepath + 'CC_Full.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')
#存储表名
    cfname = savepath + 'CC_TCName.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')

    lists = os.listdir(filepath)  # 按顺序列出文件夹下所有的目录与文件
    lists.sort(key=lambda x: int(x[:-4]))
    #print(lists)

    for list in lists:
        if 'txt' in list:
            fc.write(list + '\r\n')  # 将文件名保存至TCName文本文件中
            path = os.path.join(filepath, list)
            #print (path)#输出每个文件的地址

            if os.path.isfile(path):#如果该地址存在，读取文档中的内容
                f = codecs.open(path, 'rb',encoding='ISO-8859-1')
                lines = f.readlines()
                # 将需求或软件制品文本保存至文本文件
                for line in lines:
                    line=line.strip('\r\n')
                    fp.write(line)
                fp.write('\r\n')

# eTour文件读取（UC）
def eTourUCTranslate(filepath, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fname = savepath + 'UC_FULL.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')

    cfname = savepath + 'UC_TName.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')

    lists = os.listdir(filepath)  # 按顺序列出文件夹下所有的目录与文件
    c = []
    for list in lists:
        c.append(list.split('.')[0])
    c.sort(key=lambda x:int(x[2:]))
    print(c)

    lists_sort=[]
    for i in c:
        lists_sort.append((str(i) + '.TXT'))

    #print(lists)
    for list in lists_sort:
        print(list)
        if 'TXT' in list:
            fc.write(list + '\r\n')
            path = os.path.join(filepath, list)
            print(path)
            if os.path.isfile(path):
                lines = codecs.open(path, 'r',encoding='ISO-8859-1')
                for line in lines:
                    if 'Use case name'in line:
                        line=line.replace('Use case name','')
                    line = line.strip('\r\n')
                    fp.write(line)

            #提取第一行
            ''' flag=0
                for line in lines:
                    flag=flag+1
                    if flag==2:
                        line=line.strip('\r\n')
                        fp.write(line)'''

        fp.write('\r\n')


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
                index = 0
                num = 0
                code_num = 0

    #eTPUR提取注释
                ''' for line in lines:
                    #块注释
                    if '/ **' in line:
                        flag=1
                        if re.sub('/ \*\*', '', line).strip()!='':
                            comment.append(re.sub('/ \*\*', '', line).strip())
                    if flag==1 and '*' in line and '/ **' not in line:
                        if re.sub('\*', '', line).strip()!='':
                            comment.append(re.sub('\*', '', line).strip())
                    if '* /' in line:
                        flag=0
                    if '/ / 'in line: # 行注释
                        comment.append(re.sub('/ / ', '', line).strip())'''
                #itrust注释

                for line in lines:
                    #块注释
                    if line!='':
                        code_num = code_num+1
                    if '/*' in line and '/**' not in line:
                        flag=1
                        if re.sub('/*', '', line).strip() != '':
                            #comment.append(re.sub('/*', '', line).strip())
                            num=num+1
                    if flag == 1 and '*' in line and '/*' not in line:
                        if re.sub('\*', '', line).strip() != '' and '*/' not in line:
                            #comment.append(re.sub('\*', '', line).strip())
                            num = num + 1
                        if '*/' in line:
                            flag = 0
                            if re.sub('\*/', '', line).strip() != '':
                                #comment.append(re.sub('\*/', '', line).strip())
                                num = num + 1
                    if '/**' in line:
                        flag=1
                        if re.sub('/\*\*', '', line).strip()!='':
                            #comment.append(re.sub('/*\\*', '', line).strip())
                            num = num + 1
                    if flag==1 and '*' in line and '/**' not in line:
                        if re.sub('\*', '', line).strip()!='' and '*/'not in line:
                            #comment.append(re.sub('\*', '', line).strip())
                            num = num + 1
                        if '*/' in line:
                            flag = 0
                            if re.sub('\*/', '', line).strip()!='':
                                #comment.append(re.sub('\*/', '', line).strip())
                                num = num + 1
                    if '//'in line:
                        #comment.append(re.sub('//', '', line).strip())
                        comment.append(line.split('//')[1].strip())
                        num += num
                        #albergate
                '''for line in lines:
                    if '/*' in line:
                        flag = 1
                        if re.sub('/\*', '', line).strip() != '':
                            comment.append(re.sub('/\*', '', line).strip())
                    if flag == 1 and '*' in line and '/**' not in line:
                        if re.sub('\*', '', line).strip() != '' and '*/' not in line:
                            comment.append(re.sub('\*', '', line).strip())
                        if '*/' in line:
                            if re.sub('\*/', '', line).strip() != '':
                                comment.append(re.sub('\*/', '', line).strip())
                            flag = 0
                    if '//' in line:  # 行注释
                        line=line.split('//')
                        comment.append(re.sub('//', '', line[1]).strip())'''
                #print(comment)
                result=result+num / (code_num + num)
                print(result)
                if comment != '':
                    fc.write(file + '\n')
                    for line in comment:
                        fp.write(line+' ')
                    fp.write('\n')
    print(num)
    print(code_num)
    print(num / (code_num + num))
    print(result/98)

# eTOUR、Albergate代码文件读取javaparser（类名、方法名、变量名）
def readcode_2(inputfile, savepath):
    data = pd.read_excel(inputfile)
    #print(data)
    CN = []
    MN = []
    VN = []
    code_name=[]
    n=0
    #按行处理
    for num, entity in data.iterrows():
       if '包名' in entity[1]:
           CN.append('换行')
           MN.append('换行')
           VN.append('换行')
           code_name.append(entity[0])
           n=n+1
       elif '类名' in entity[1]:
           CN.append(entity[0])
       elif '域名' in entity[1]:
           VN.append(entity[0])
       elif '方法名' in entity[1]:#方法名
           MN.append(entity[0])
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    #存储文件
    print(CN)
    print(MN)
    print(VN)
    print(n)
    CNname = savepath + 'CN.txt'
    fc = codecs.open(CNname, 'a+', 'utf-8')

    MNfname = savepath + 'MN.txt'
    fm = codecs.open(MNfname, 'a+', 'utf-8')

    VNfname = savepath + 'VN.txt'
    fv = codecs.open(VNfname, 'a+', 'utf-8')

    codefname = savepath + 'Code_TName.txt'
    ff = codecs.open(codefname, 'a+', 'utf-8')
    flag=0
    for cn in CN:
        if '换行' in cn:
            if flag==1:
                fc.write('\r\n')
            else:
                flag=1
        else:
            fc.write(cn+' ')
    flag = 0
    for cn in MN:
        if '换行' in cn:
            if flag == 1:
                fm.write('\r\n')
            else:
                flag = 1
        else:
            fm.write(cn+' ')

    flag = 0
    for cn in VN:
        if '换行' in cn:
            if flag == 1:
                fv.write('\r\n')
            else:
                flag = 1
        else:
            fv.write(cn+' ')

    for i in code_name:
        ff.write(i+'\r\n')


# EBT文件读取(requirements->testcase)
def EBT_read(inputfile, savepath):
    if not os.path.exists(savepath) and os.path.isfile(inputfile):
        os.makedirs(savepath)

    fname = savepath + 're.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')

    cfname = savepath + 're_TName.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')

    #读取文件
    f = codecs.open(inputfile, 'rb', 'utf-8')
    lists=f.readlines()

    #test case
    '''num=141
    for list in lists:
        if str(num) in list:
            if str(num)+' Test case: 'in list:
                list=list.replace(str(num)+' Test case: ','')
            elif str(num)+' "Test case: 'in list:
                list = list.replace(str(num)+' "Test case: ', '')

            fc.write(str(num)+'\r\n')
            fp.write(list)
            num=num+1'''
    #requirement
    num=100
    for list in lists:
        if str(num) in list:
            list = list.replace(str(num)+'	', '')
            fc.write(str(num) + '\r\n')
            fp.write(list)
            num = num + 1
    fc.close()
    fp.close()

#iTrust
def iTrustTranslate(filepath,savepath):
    # file_dir = 'E:/ProjectTest/new/IceBreaker/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fname = savepath + 'UC_Full.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')
#存储表名
    cfname = savepath + 'UC_TCName.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')

    lists = os.listdir(filepath)  # 按顺序列出文件夹下所有的目录与文件
    c = []
    for list in lists:
        c.append(list.split('.')[0])
    c.sort(key=lambda x: int(x[2]))
    print(c)

    lists_sort = []
    for i in c:
        lists_sort.append((str(i) + '.TXT'))
    #lists.sort(key=lambda x: int(x[:-4]))
    #print(lists)

    for list in lists:
        if 'txt' in list:
            fc.write(list + '\r\n')  # 将文件名保存至TCName文本文件中
            path = os.path.join(filepath, list)
            #print (path)#输出每个文件的地址

            if os.path.isfile(path):#如果该地址存在，读取文档中的内容
                f = codecs.open(path, 'rb',encoding='ISO-8859-1')
                lines = f.readlines()
                # 将需求或软件制品文本保存至文本文件
                for line in lines:
                    line=line.strip('\r\n')
                    fp.write(line)
                fp.write('\r\n')



#Albergate(re)
def AlbergateTranslate(filepath,savepath):
    # file_dir = 'E:/ProjectTest/new/IceBreaker/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fname = savepath + 'low_Full.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')
#存储表名
    cfname = savepath + 'low_TCName.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')

    lists = os.listdir(filepath)  # 按顺序列出文件夹下所有的目录与文件
    for list in lists:
        if 'txt' in list:
            fc.write(list + '\r\n')  # 将文件名保存至TCName文本文件中
            path = os.path.join(filepath, list)
            #print (path)#输出每个文件的地址

            if os.path.isfile(path):#如果该地址存在，读取文档中的内容
                f = codecs.open(path, 'rb',encoding='ISO-8859-1')
                lines = f.readlines()
                # 将需求或软件制品文本保存至文本文件
                for line in lines:
                    line=line.strip('\r\n')
                    fp.write(line)
                fp.write('\r\n')
#SMOS(re)
def SMOS_Translate(filepath,savepath):
    # file_dir = 'E:/ProjectTest/new/IceBreaker/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fname = savepath + 'UC_content.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')
#存储表名
    cfname = savepath + 'UC_content_TCName.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')

    lists = os.listdir(filepath)  # 按顺序列出文件夹下所有的目录与文件
    c = []
    for list in lists:
        list=list.split('.')[0]
        c.append(int(list.split('S')[2]))
    c.sort()
    print(c)
    lists_sort = []
    for i in c:
        lists_sort.append('SMOS'+str(i) + '.txt')
    print(lists_sort)
    for list in lists_sort:
        if 'txt' in list:
            fc.write(list + '\r\n')  # 将文件名保存至TCName文本文件中
            path = os.path.join(filepath, list)
            #print (path)#输出每个文件的地址

            if os.path.isfile(path):#如果该地址存在，读取文档中的内容
                f = codecs.open(path, 'rb',encoding='utf-8')
                lines = f.readlines()
                # 将需求或软件制品文本保存至文本文件
                for line in lines:
                    if 'Nome' in line:
                        line=line.replace('Nome','')
                    elif 'Attori' in line:
                        line = line.replace('Attori', '')
                    elif 'Descrizione' in line:
                        line = line.replace('Descrizione:', '')
                    elif 'Precondizioni' in line:
                        line = line.replace('Precondizioni:', '')
                    elif 'Sistema' in line:
                        line = line.replace('Sistema', '')
                    elif 'Postcondizioni' in line:
                        line = line.replace('Postcondizioni', '')
                    line=line.strip('\r\n')
                    fp.write(line+' ')
                fp.write('\r\n')


if __name__ == '__main__':
    #filepath = '../CoESTData/WARC/SRS/'
    #savepath = '../sample-Data/WARC/'
    #CodeTranslate(filepath, savepath)

    #filepath = '../CoESTData/eTOUR/UC/'
   # savepath = '../sample-Data/eTOUR/'
    #eTourUCTranslate(filepath, savepath)

    # easyClinic文件
    #filepath = '../CoESTData/EasyClinic/EasyClinic/EasyClinicDataset/2 - docs (English)/1 - use cases/'
    #filepath = '../CoESTData/EasyClinic/EasyClinic/EasyClinicDataset/2 - docs (English)/3 - test cases/'
    #filepath = '../CoESTData/EasyClinic/EasyClinic/EasyClinicDataset/2 - docs (English)/2 - Interaction diagrams/'
    #filepath = '../CoESTData/EasyClinic/EasyClinic/EasyClinicDataset/2 - docs (English)/4 - class description/'
    #savepath = '../sample-Data/EasyClinic/'
    #EasyClinicTranslate(filepath, savepath)


    #filepath = '../CoESTData/Albergate/to_be_traced_source_code/'
    #savepath = '../sample-Data/Albergate/'
    #readcode_1(filepath, savepath)

    #filepath = '../sample-Data/Albergate/code.xlsx'
   # savepath = '../sample-Data/Albergate/'
   # readcode_2(filepath, savepath)

   # filepath = '../CoESTData/eTOUR/CC/'
   # savepath = '../sample-Data/eTOUR/'
    #filepath = '../CoESTData/iTrust/iTrust-code'
   # savepath = '../sample-Data/iTrust/'
   # readcode_1(filepath, savepath)

    #filepath = '../CoESTData/EBT/EBT/EBT/Requirements.txt'#Testcases
   # savepath = '../sample-Data/EBT/'
   # EBT_read(filepath, savepath)

    #filepath = '../CoESTData/iTrust/iTrust/UC/'  # Testcases
    #savepath = '../sample-Data/iTrust/'
    #iTrustTranslate(filepath, savepath)
    #filepath = '../CoESTData/GANNT/high/'
    #savepath = '../sample-Data/GANNT/'
   # AlbergateTranslate(filepath, savepath)
   # filepath = '../CoESTData/SMOS/uc/'
    #savepath = '../sample-Data/SMOS_2/'
    #SMOS_Translate(filepath, savepath)
    filepath = '../CoESTData/SMOS/cc(English)/'
    savepath = '../sample-Data/new/'
    '''filepath = '../CoESTData/iTrust/iTrust-code/'
    savepath = '../sample-Data/new/'''''
    readcode_1(filepath, savepath)

