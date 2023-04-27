#-*- coding: utf-8 -*-
import codecs
import pandas as pd
import os
#合并不同的代码文件
def MNandCMT(output_path,MNfname,CMTfname):

    if os.path.isfile(MNfname) and os.path.isfile(CMTfname):
        mn = codecs.open(MNfname, 'rb', 'utf-8')
        lines_mn = mn.readlines()
        cmt = codecs.open(CMTfname, 'rb', 'utf-8')
        lines_cmt = cmt.readlines()
        all=[]
        for i in range(len(lines_mn)):
            #print(lines_mn[i])
            if lines_mn[i].strip('\n')=='':
                all.append(lines_cmt[i].strip('\r\n'))
            else:
                all.append(lines_mn[i].strip('\r\n')+' '+lines_cmt[i].strip('\r\n'))
    with open(output_path, 'w',encoding='utf-8')as fp:
        for i in all:
            fp.write(i+'\n')
if __name__ == '__main__':
    #inputfile = '../sample-Data/iTrust/code1.xlsx'
    #savepath = '../sample-Data/iTrust/'
    #readcode_2(inputfile, savepath)
    '''MNfname = '../sample-Data/new/CMT_num.txt'
    CMTfname = '../sample-Data/new/CMT.txt'
    output_path = '../sample-Data/new/CMT2.txt'
    MNandCMT(output_path, MNfname, CMTfname)'''
    MNfname = '../sample-Data/new/CN.txt'
    CMTfname = '../sample-Data/new/CMT.txt'
    output_path = '../sample-Data/new/CN_CMT.txt'
    MNandCMT(output_path, MNfname, CMTfname)

    MNfname = '../sample-Data/new/MN.txt'
    CMTfname = '../sample-Data/new/CN.txt'
    output_path = '../sample-Data/new/CN_MN.txt'
    MNandCMT(output_path,MNfname,CMTfname)

    MNfname = '../sample-Data/new/VN.txt'
    CMTfname = '../sample-Data/new/CN.txt'
    output_path = '../sample-Data/new/CN_VN.txt'
    MNandCMT(output_path, MNfname, CMTfname)

    MNfname = '../sample-Data/new/MN.txt'
    CMTfname = '../sample-Data/new/VN.txt'
    output_path = '../sample-Data/new/MN_VN.txt'
    MNandCMT(output_path, MNfname, CMTfname)

    MNfname = '../sample-Data/new/MN.txt'
    CMTfname = '../sample-Data/new/CMT.txt'
    output_path = '../sample-Data/new/MN_CMT.txt'
    MNandCMT(output_path, MNfname, CMTfname)

    MNfname = '../sample-Data/new/CMT.txt'
    CMTfname = '../sample-Data/new/VN.txt'
    output_path = '../sample-Data/new/VN_CMT.txt'
    MNandCMT(output_path, MNfname, CMTfname)

    MNfname = '../sample-Data/new/CN_MN.txt'
    CMTfname = '../sample-Data/new/VN.txt'
    output_path = '../sample-Data/new/CN_MN_VN.txt'
    MNandCMT(output_path, MNfname, CMTfname)
    
    MNfname = '../sample-Data/new/CN_VN.txt'
    CMTfname = '../sample-Data/new/CMT.txt'
    output_path = '../sample-Data/new/CN_VN_CMT.txt'
    MNandCMT(output_path, MNfname, CMTfname)

    MNfname = '../sample-Data/new/CN_MN.txt'
    CMTfname = '../sample-Data/new/CMT.txt'
    output_path = '../sample-Data/new/CN_MN_CMT.txt'
    MNandCMT(output_path, MNfname, CMTfname)
    
    MNfname = '../sample-Data/new/MN_VN.txt'
    CMTfname = '../sample-Data/new/CMT.txt'
    output_path = '../sample-Data/new/MN_VN_CMT.txt'
    MNandCMT(output_path, MNfname, CMTfname)

    MNfname = '../sample-Data/new/CN_MN.txt'
    CMTfname = '../sample-Data/new/VN_CMT.txt'
    output_path = '../sample-Data/new/CN_MN_VN_CMT.txt'
    MNandCMT(output_path, MNfname, CMTfname)