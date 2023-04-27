import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import codecs
#打开并读取文件
import nltk
def Data_clear(path,output_path):
    if os.path.isfile(path):
        f = codecs.open(path, 'rb', 'utf-8')
        lines = f.readlines()
        filtered = []
    for text_list in lines:
        # 1，分词
        text_list = nltk.word_tokenize(text_list)
        # 2，去除标点符号和停用词
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','<','>','/','@','»','¿','b','p','©','â','¢','’','•','“','”',"==","="]
        text_list = [word for word in text_list if word not in english_punctuations]
        stops = set(stopwords.words("english"))
        text_list = [word for word in text_list if word not in stops]
        # 3，词性标注并保留动词和名词
        text_list = nltk.pos_tag(text_list)
        text_list = [name for name,value in text_list if value in ['NN','NNP','NNPS','NNS','VB','VBD','VBG','VBN','VBP','VBZ']]
        filtered=filtered+text_list
        filtered.append('\n')
        #print(type(filtered))
    # 4，词干提取并写入文件
    with open(output_path, 'w',encoding='utf-8')as fp:
        for i in filtered:
            #porter_stemmer = PorterStemmer()
            snowballStemmer=SnowballStemmer('english')
            if '\n' in i:
                fp.write(snowballStemmer.stem(i))
            else:
                fp.write(snowballStemmer.stem(i)+' ')
                #print(i)
    fp.close()



if __name__ == '__main__':
    path = 'Pig/CN_MN_VN_CMT.txt'
    output_path = 'Pig/CN_MN_VN_CMT_clear.txt'
    Data_clear(path, output_path)




