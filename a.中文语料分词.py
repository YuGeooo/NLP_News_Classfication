# -*- coding: utf-8 -*-
"""
@version: python3.8.5
@author: GYQ
@file: a.中文语料分析.py

前言：在完成本题的过程中，我学习和参考了博文《Python中文文本分类》以及慕课《Python机器学习应用》；
      停用字库来自四川大学机器智能实验室；新闻下载来自于《联合早报》《人民日报》《湖北日报》《中国环境新闻网》等
    https://blog.csdn.net/github_36326955/article/details/54891204
    https://www.icourse163.org/course/BIT-1001872001
"""

import os # python内置的包，用于进行文件目录操作
import jieba

defultloc = "C:/Users/GYQ/Desktop/Python"

# 保存至文件
def _savefile(savepath, content):
    with open(savepath, "w", encoding = 'utf-8') as fp:
        fp.write(content)
        
# 读取文件
def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

def corpus_segment(corpus_path, seg_path):
    '''corpus_path 是未分词语料库路径
          seg_path 是分词后语料库存储路径'''
    catelist = os.listdir(corpus_path)  # 获取corpus_path下的所有子目录

    # 获取每个目录（类别）下所有的文件
    for mydir in catelist:
        '''
        这里mydir就是/src_dir_test/环境类新闻/xxx.txt中的"环境类新闻"（即catelist中的一个类别）
        '''
        class_path = corpus_path + mydir + "/"                    # 拼出分类子目录的路径
        seg_dir = seg_path + mydir + "/"                          # 拼出分词后存贮的对应目录路径
 
        if not os.path.exists(seg_dir):                           # 是否存在分词目录，如果没有则创建该目录
            os.makedirs(seg_dir)
 
        file_list = os.listdir(class_path)                        # 获取未分词语料库中某一类别中的所有文本
     
        for file_path in file_list:                               # 遍历类别目录下的所有文件
            fullname = class_path + file_path                     # 拼出文件名全路径
            content = _readfile(fullname)                         # 读取文件内容
            content = content.decode('utf-8')
            '''
            此时，content里面存储的是原文本的所有字符，例如多余的空格、空行、回车等等，
            这里需变成只有标点符号做间隔的紧凑的文本内容
            ''' 
            content = content.replace("\r\n", "")                 # 删除换行
            content = content.replace(" ", "")                    # 删除空行、多余的空格
            content_seg = jieba.cut(content)                      # 为文件内容分词
            _savefile(seg_dir + file_path, " ".join(content_seg)) # 将处理后的文件保存到分词后语料目录
 
    print("中文语料分词结束！！！")


if __name__=="__main__":
    
    #对训练集进行分词
    corpus_path = defultloc + "/第八题训练集/src_dir_test/"      # 非环境类未分词分类语料库路径
    seg_path = defultloc + "/第八题训练集/已分词/"               # 非环境类分词后分类语料库路径
    corpus_segment(corpus_path, seg_path)

    #对测试集进行分词
    corpus_path = defultloc + "/测试集/src_dir_test/"      
    seg_path = defultloc + "/测试集/已分词/"                       
    corpus_segment(corpus_path,seg_path)

