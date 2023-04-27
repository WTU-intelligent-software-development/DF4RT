#!/usr/bin/python
# -*- coding:utf-8 -*-
from xml.dom import minidom
import os
import codecs
#处理iTrustxml文件
def iTrust_xml():
    # 导入模块
    dom=minidom.parse("D:\\pycharmprojects\\venv\\CoESTData\\iTrust\\iTrust\\answer_req_javacode.xml") #2.加载xml文件
    Requirements=dom.getElementsByTagName("source_artifact_id") #获取节点列表
    Code=dom.getElementsByTagName("target_artifact_id")
    #存入文件
    with open('../sample-data/iTrust/answer_req_javacode.txt','w') as fp:
        for i in range(len(Requirements)):
            fp.write(Requirements[i].firstChild.data+' '+Code[i].firstChild.data+'\n')   #打印节点数据



#处理CM1xml文件
def CM1_xml_1():
    # 导入模块
    dom=minidom.parse("D:\\pycharmprojects\\venv\\CoESTData\\CM1-NASA\\CM1-targetArtifacts.xml") #2.加载xml文件
    Requirements=dom.getElementsByTagName("id") #获取节点列表
    Code=dom.getElementsByTagName("content")
    savepath='../sample-data/CM1/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fname = savepath + 'targetArtifacts_name.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')
    cfname = savepath + 'targetArtifacts.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')
    for i in range(len(Requirements)):
        fp.write(Requirements[i].firstChild.data+'\n')
        fc.write(Code[i].firstChild.data.replace('\n','')+'\n')  # 打印节点数据

def CM1_xml_2():
    # 导入模块
    dom=minidom.parse("D:\\pycharmprojects\\venv\\CoESTData\\CM1-NASA\\CM1-sourceArtifacts.xml") #2.加载xml文件
    Requirements=dom.getElementsByTagName("id") #获取节点列表
    Code=dom.getElementsByTagName("content")
    savepath='../sample-data/CM1/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fname = savepath + 'sourceArtifacts_name.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')
    cfname = savepath + 'sourceArtifacts.txt'
    fc = codecs.open(cfname, 'a+', 'utf-8')
    for i in range(len(Requirements)):
        fp.write(Requirements[i].firstChild.data+'\n')
        fc.write(Code[i].firstChild.data)  # 打印节点数据

def CM1_xml_3():
    # 导入模块
    dom=minidom.parse("D:\\pycharmprojects\\venv\\CoESTData\\CM1-NASA\\CM1-answerSet.xml") #2.加载xml文件
    Requirements=dom.getElementsByTagName("source_artifact_id") #获取节点列表
    Code=dom.getElementsByTagName("target_artifact_id")
    savepath='../sample-data/CM1/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fname = savepath + 'CM1-answerSet.txt'
    fp = codecs.open(fname, 'a+', 'utf-8')
    for i in range(len(Requirements)):
        fp.write(Requirements[i].firstChild.data+' '+Code[i].firstChild.data+'\n') # 打印节点数据

if __name__ == '__main__':
    CM1_xml_3()

