# -*- coding: utf-8 -*-
"""
@version: python3.8.5
@author: GYQ
@file: d.分类器.py
"""

import os
import _pickle as pickle
from sklearn.tree import DecisionTreeClassifier # 导入决策树分类器
from sklearn import metrics
import shutil
 
defultloc = "C:/Users/GYQ/Desktop/Python"
 
# 读取bunch对象
def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch
 
# 导入训练集
trainpath = defultloc + "/第八题训练集/tfdifspace.dat"
train_set = _readbunchobj(trainpath)
 
# 导入测试集
testpath = defultloc + "/测试集/testspace.dat"
test_set = _readbunchobj(testpath)
 
# 训练分类器
dt = DecisionTreeClassifier().fit(train_set.tdm, train_set.label)

# 预测分类结果
predicted = dt.predict(test_set.tdm)

print("\n>>>预测正确：")
for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel == expct_cate:
        print(file_name, ": 实际类别:", flabel, " ->  预测类别:", expct_cate)
        
print("\n>>>预测错误：")
for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " ->  预测类别:", expct_cate)
        
print("\n----------预测完毕----------")

# 计算分类精度：
def metrics_result(actual, predict):
    print('准确度:{0:.3f}'.format(metrics.accuracy_score(actual, predict)))
    print('精确率:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回率:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
 
metrics_result(test_set.label, predicted)

# 按预测的标签复制文件到指定目录
for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):

    file_path = file_name
    (filepath,tempfilename) = os.path.split(file_path)
    (filename,extension) = os.path.splitext(tempfilename)

    if not os.path.exists(defultloc + "/自动分类_测试集新闻/" + expct_cate + "/"):                    
        os.makedirs(defultloc + "/自动分类_测试集新闻/" + expct_cate + "/")
        
    shutil.copyfile(file_path, defultloc + "/自动分类_测试集新闻/" + expct_cate + "/" + filename + extension)