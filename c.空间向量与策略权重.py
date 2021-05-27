# -*- coding: utf-8 -*-
"""
@version: python3.8.5
@author: GYQ
@file: c.空间向量与策略权重.py
"""

from sklearn.datasets.base import Bunch
import _pickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer

defultloc = "C:/Users/GYQ/Desktop/Python"

# 读取文件
def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content
 
# 读取bunch对象
def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

# 写入bunch对象
def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)
 
# 这个函数用于创建TF-IDF词向量空间
def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path = None):
    '''
        stopword_path: 停用词表路径
           bunch_path: bunch路径
           space_path: 创建后的词向量空间路径
     train_tfidf_path: 将测试数据映射到训练集的TF-IDF词向量空间上之后的路径，创建词向量空间时不调用
    '''
    stpwrdlst = _readfile(stopword_path).splitlines() # 读取停用词
    bunch = _readbunchobj(bunch_path)                 # 导入分词后的词向量bunch对象

    # 构建tf-idf词向量空间对象
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})
    '''
        tdm用于存放计算后得到的TF-IDF权重矩阵
        vocabulary用于存放词典索引
    '''
    if train_tfidf_path is not None:
        
        # 导入训练集的TF-IDF词向量空间
        trainbunch = _readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary

        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        # 使用TfidfVectorizer初始化向量空间模型
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        '''
         关于参数：
        
           stop_words: 传入停用词，以后我们获得vocabulary_的时候，就会根据文本信息去掉停用词得到

         sublinear_tf: 计算tf值采用亚线性策略。比如，我们以前算tf是词频，现在用1+log(tf)来充当词频。

           smooth_idf: 计算idf的时候log(分子/分母)分母有可能是0，smooth_idf会采用log(分子/(1+分母))的方式解决。默认已经开启，无需关心。

                 norm: 归一化，我们计算TF-IDF的时候，是用TF*IDF，TF可以是归一化的，也可以是没有归一化的，一般都是采用归一化的方法，默认开启.

               max_df: 有些词，他们的文档频率太高了（一个词如果每篇文档都出现，那还有必要用它来区分文本类别吗？当然不用了呀），所以，我们可以
                       设定一个阈值，比如float类型0.5（取值范围[0.0,1.0]）,表示这个词如果在整个数据集中超过50%的文本都出现了，那么我们也把它列
                       为临时停用词。当然你也可以设定为int型，例如max_df=10,表示这个词如果在整个数据集中超过10的文本都出现了，那么我们也把它列
                       为临时停用词。

               min_df: 与max_df相反，虽然文档频率越低，似乎越能区分文本，可是如果太低，例如10000篇文本中只有1篇文本出现过这个词，仅仅因为这1篇
                       文本，就增加了词向量空间的维度，太不划算。当然，max_df和min_df在给定vocabulary参数时，就失效了。
        '''
        # 此时tdm里面存储的就是if-idf权值矩阵
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_
 
    _writebunchobj(space_path, tfidfspace)
    print("if-idf词向量空间实例创建成功！！！")
 
if __name__ == '__main__':
    stopword_path = defultloc + "/scu_stopwords.txt"                     # 停用词表的路径
    bunch_path = defultloc + "/第八题训练集/train_set.dat"                # 导入训练集Bunch的路径
    space_path = defultloc + "/第八题训练集/tfdifspace.dat"               # 训练集词向量空间保存路径
    vector_space(stopword_path, bunch_path, space_path)                  # 不传入第四个参数

    bunch_path = defultloc + "/测试集/test_set.dat"                       # 导入测试集Bunch的路径
    space_path = defultloc + "/测试集/testspace.dat"                      # 测试集词向量空间保存路径
    train_tfidf_path = defultloc + "/第八题训练集/tfdifspace.dat" 
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path) # 传入了第四个参数，将测试数据映射到训练集的TF-IDF词向量空间上
