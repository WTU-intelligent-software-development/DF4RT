#!/usr/bin/python
# -*-coding:utf-8-*-
#Number_unique_terms_in_document1
from collections import Counter
import numpy as np
import sys
sys.path.append('Set_generation.py')
import Set_generation
from gensim import corpora
from createCorpusFromDocumentList import  createCorpusFromDocumentList
def calculateUniqueWordCount(termList):
    try:
        uniqueWordCount = len(list(Counter(termList).keys()))
    except:
        uniqueWordCount = np.nan
    print(uniqueWordCount)

    return(uniqueWordCount)

def calculateTotalWordCount(termList):
    try:
        totalWordCount = len(termList)
    except:
        totalWordCount = np.nan

    return(totalWordCount)

def calculateOverlapBetweenDocuments(termList1, termList2, comparisonList):
    if(isinstance(termList1, list) & isinstance(termList2, list)):
        set1 = set(termList1)
        set2 = set(termList2)
        overlap = set1 & set2
        union = set1 | set2
  
        #Compare the overlap to list1
        if(comparisonList == 'list1'):
            percentageOverlap = float(len(overlap)) / len(set1) * 100
        #Compare the overlap to list2
        elif(comparisonList == 'list2'):
            if (len(set2)>0):
                percentageOverlap = float(len(overlap)) / len(set2) * 100
            else:
                percentageOverlap = float(0)
        elif(comparisonList == 'union'):
            percentageOverlap = float(len(overlap)) / len(union) * 100
        return(percentageOverlap)

    else:
        return(np.nan)


query_file = "../iTrust/UC_clear.txt"
queried_file = "../iTrust/code_feature/CN_MN_VN_CMT_clear.txt"
queried_line = Set_generation.set_generation(queried_file)
query_line = Set_generation.set_generation(query_file)
for i in query_line:
    for j in queried_line:

         print(calculateUniqueWordCount(i+j))
# for i in query_line:
#     for j in queried_line:
#         calculateOverlapBetweenDocuments(i, j, "list1")
#    calculateUniqueWordCount(i)





