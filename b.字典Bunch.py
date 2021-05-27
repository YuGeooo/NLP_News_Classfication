# -*- coding: utf-8 -*-
"""
@version: python3.8.5
@author: GYQ
@file: b.字典Bunch.py
"""

import os
import _pickle as pickle
from sklearn.datasets.base import Bunch

defultloc = "C:/Users/GYQ/Desktop/Python"

# 读取文件
def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

def corpus2Bunch(wordbag_path,seg_path):
    catelist = os.listdir(seg_path)                     # 获取seg_path下的所有子目录，也就是分类信息

    # 创建一个Bunch
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    '''
     在Bunch里有4个成员：
         target_name：是一个list，存放的是整个数据集的类别集合
               label：是一个list，存放的是所有文本的标签
           filenames：是一个list，存放的是所有文本文件的名字
            contents：是一个list，分词后文本文件（一个文本文件只有一行）
    ''' 
    bunch.target_name.extend(catelist)
    # 获取每个目录下所有的文件
    for mydir in catelist:
        class_path = seg_path + mydir + "/"             # 拼出分类子目录的路径
        file_list = os.listdir(class_path)              # 获取class_path下的所有文件
        for file_path in file_list:                     # 遍历类别目录下文件
            fullname = class_path + file_path           # 拼出文件名全路径
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(_readfile(fullname))  # 读取文件内容

    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print ("构建文本对象结束！！！")
 
if __name__ == "__main__":
    # 对训练集进行Bunch化操作：
    wordbag_path = defultloc + "/第八题训练集/train_set.dat"   # Bunch存储路径
    seg_path = defultloc + "/第八题训练集/已分词/"             # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)

    # 对测试集进行Bunch化操作：
    wordbag_path = defultloc + "/测试集/test_set.dat"         # Bunch存储路径
    seg_path = defultloc + "/测试集/已分词/"                  # 测试集_非环境类分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)