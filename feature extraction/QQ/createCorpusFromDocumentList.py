#!/usr/bin/python
# -*-coding:utf-8-*-
import sys
sys.path.append('Set_generation.py')
import Set_generation
#生成语料库,转换成sklearn可以使用的形式
def createCorpusFromDocumentList(token_column):
    #token_list = token_column.tolist()
    token_list = token_column
    corpus_list=[]
    for document in token_list:
        #Only join to the string when a list. When it is not a list, then it is np.NaN, thus no changes
        if(isinstance(document, list)):
            #Transform list to a string for SKLEARN to accept the input.
            token_string = ' '.join(document)
            #Add string to the corpus list
            corpus_list.append(token_string)

    return(corpus_list)


if __name__ == '__main__':
    fname = "../iTrust/code_feature/CN_MN_VN_CMT_clear.txt"
    tname = "../iTrust/UC_clear.txt"
    output_fname = "../iTrust/QQ"
    #queried_line = Set_generation.set_generation(fname)
    query_line = Set_generation.set_generation(tname)
    createCorpusFromDocumentList(query_line)