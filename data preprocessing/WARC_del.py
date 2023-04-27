#!/usr/bin/python 
# -*- coding:utf-8 -*-
import codecs
import re
import chardet

#去掉WARC的里面的FR和SRS
def WARCresult(filepath):
    f = codecs.open(filepath, 'rb')
    truelines = f.readlines()


    FRS = []
    SRS = []
    for trueline in truelines:
        encode_type = chardet.detect(trueline)
        trueline = trueline.decode(encode_type['encoding'])
        if '%' not in trueline:
            s = []
            words = trueline.split('\t')
            for word in words:
                if 'FR' in word:
                    FRS.append(re.sub('\D', '', word))
                else:
                    w = word.split(' ')
                    for i in w:
                        if 'SRS'in i:
                            s.append(re.sub('\D', '', i))
            SRS.append(s)
    print (FRS)
    print (SRS)

def IceBreakresult(filepath):
    f = codecs.open(filepath, 'rb')
    truelines = f.readlines()
    ReToClass = {}
    Re = []
    Class = []
    for trueline in truelines:
        words = trueline.decode('utf-8').split(' ')
        ReToClass[(int(re.sub('\D', '', words[0])), int(re.sub('\D', '', words[1])))] = 1

    repath = '../sample-data/WARC/FRS_1.txt'
    classpath = '../sample-data/WARC/FRS_2.txt'
    f1 = codecs.open(repath, 'rb')
    requirements = f1.readlines()
    for req in requirements:
        r = req.split(' ', 1)
        Re.append(r[0])

    f2 = codecs.open(classpath, 'rb')
    classes = f2.readlines()
    for cla in classes:
        c = cla.split(' ', 1)
        Class.append(c[0])
    print (Re[187])
    print (Class[53])
    print (ReToClass)


if __name__ == '__main__':
    filepath = '../sample-data/WARC/FRS.txt'
    WARCresult(filepath)