#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:20:57 2019

@author: daniel
"""
# 面向pandas的数据类型，onehotencoder，前提：数据类型，且是数字化的，否则先要转化成数字
"""转成数字,例如y数据为 yes, no,经过以下数据处理后，变成了1,0等序列，然后再使用one-hot-encoder
labelencoder_y = LabelEncoder()
y = labelencodder_y.fit_transform()
"""
"""
# 多列数据需要one-hot以及拼接到原有的数据当中.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pandas.core.frame import DataFrame
# 需要one-hot的列
dummy_fields = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
# 取第0列
onehotencoder = OneHotEncoder(categorical_features = [0])
for each in dummy_fields:
    # 读取列出来（变成了一维的向量）
    each_one_hot = rides[each].values
    # 将行向量变成纵向量
    each_one_hot = each_one_hot.reshape(len(each_one_hot), -1)
    # one-hot
    each_one_hot = onehotencoder.fit_transform(each_one_hot).toarray()
    # 获取each_one_hot列数，将标号与name赋值给columnname
    columnname = [each+str(i) for i in range(each_one_hot.shape[1])]
    # 将each列one-hot结果加上 columname标题，然后转为DataFrame类型
    each_one_hot = DataFrame(each_one_hot, columns = columnname)
    # 将each_one_hot添加到 由于读取的rides数据中
    rides = pd.concat([rides, each_one_hot], axis = 1)
"""


"""
# 模型的保存和加载， one-hot, label-encoding及其还原
"""
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

def save_model(model, path = 'model.pkl'):
    import pickle
    F = open(path, 'wb')
    pickle.dump(model, F)
    F.close()

def load_model(path = None):
    if path == None:
        print("please input path of model")
    else:
        import pickle
        F = open(path, 'rb')
        model = pickle.load(F)
        F.close()
        return model

"""
one-hot, Demo
"""

a = np.array([1,2,3,1,3,2])
# 将 a 转为列向量
a = a.reshape(len(a), -1)
ohe = OneHotEncoder()
ohe.fit(a)
save_model(ohe, 'ohe.pkl')
ohe1 = load_model('ohe.pkl')
a_hot = ohe1.transform(a).toarray()


"""
# label-encoding，并逆向还原标签, Demo
"""
a = ['blue','green','white','blue']
lbe = LabelEncoder()
lbe.fit(a)

save_model(lbe, 'lbe_fit.pkl')
lbe1 = load_model('lbe_fit.pkl')
# 使用加载的ohe1函数 transform新的数据，使之按统一的函数变成one-hot模型, 以下a_hot是a的one_hot化结果
a_lbe = lbe1.transform(a)
# 还原数据
list_a = list(lbe1.inverse_transform(a_lbe))

