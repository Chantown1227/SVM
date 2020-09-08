# -*- coding = utf-8 -*-
# @time:2020/9/8 11:03
# Author:TC
# @File:支持向量机.py
# @Software:PyCharm

#SVM
# 惩罚系数C，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
# 当C趋于无穷大时，这个问题也就是不允许出现分类误差的样本存在，那这就是一个hard-margin SVM问题（过拟合）
# 当C趋于0时，我们不再关注分类是否正确，只要求间隔越大越好，那么我们将无法得到有意义的解且算法不会收敛。（欠拟合）

# gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。
# 注意sigma和gamma的关系。 gamma会影响每个支持向量对应的高斯作用范围，从而影响泛化性能。如果gamma设太大，sigma会很小，sigma很小的高斯分布长得又高又瘦，会造成只会作用于支持向量样本附近，对于未知样本分类效果很差，存在训练准确率可以很高，(如果让sigma无穷小，则理论上，高斯核的SVM可以拟合任何非线性数据，但容易过拟合)而测试准确率不高的可能，就是通常说的过拟合；而如果设的过小，则会造成平滑效应太大，无法在训练集上得到特别高的准确率，也会影响测试集的准确率，就会发生欠拟合。

# 数据EDA
# EDA:Exploratory Data Analysis探索性数据分析

import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#使用pd.read_csv从互联网读取指定数据
df_data = pd.read_csv('cancer.csv')
print(df_data.head())
print(df_data.info())

print(df_data.describe())

#标准化标签，将标签值统一转换成range(标签值个数-1)范围内
le=preprocessing.LabelEncoder() #类别特征向量化
le.fit(df_data['diagnosis'])
df_data['diagnosis']=le.transform(df_data['diagnosis'])
print(df_data['diagnosis'].head())

# DataFrame.filter(items=None, like=None, regex=None, axis=None)
#items对列进行筛选#regex表示用正则进行匹配#like进行筛选#axis=0表示对行操作，axis=1表示对列操作
df_data_X=df_data.filter(regex='_mean')
df_data_Y=df_data['diagnosis']

# center色彩两极分界线
# fmt：format的缩写，设置数值的格式化形式
# annot: 默认为False，为True的话，会在格子上显示数字
# linewidths：控制每个小方格之间的间距
# linecolor：控制分割线的颜色
# cbar_kws：关于颜色带的设置
# mask：传入布尔型矩阵，若为矩阵内为True，则热力图相应的位置的数据将会被屏蔽掉（常用在绘制相关系数矩阵图）
# vmax, vmin: 热力图颜色取值的最大值，最小值，默认会从data中推导
sns.heatmap(df_data[df_data_X.columns].corr(),linewidths=0.1,linecolor='white',annot=True,square=True,vmax=1.0,cmap=sns.color_palette('RdBu',n_colors=256))
plt.title('the feature of corr')
plt.show()

df_data_X=df_data_X.drop(['radius_mean','area_mean'],axis=1)
scalar=preprocessing.StandardScaler()
df_data_X=scalar.fit_transform(df_data_X)
print("归一化",df_data_X)

X_train,X_test,y_train,y_test=train_test_split(df_data_X,df_data_Y,test_size=0.2)

# LinearSVC是线性分类器，用于处理线性分类的数据，且只能使用线性核函数。SVC是非线性分类器，即可以使用线性核函数进行线性划分，也可以使用高维核函数进行非线性划分。
model=svm.LinearSVC()
model.fit(X_train,y_train)
pred_label=model.predict(X_test)
print('准确率：',accuracy_score(pred_label,y_test))

##网格搜索
parameters={
    'gamma':np.linspace(0.0001,0.1), #默认50个参数
    'kernel':['linear','poly','rbf','sigmoid']
}
model=svm.SVC()
#网格搜索寻优
grid_model=GridSearchCV(model,parameters,cv=10,return_train_score=True)
#cv为交叉验证（Cross Validation）,将原始数据分成K组（一般是均分），将每个子集数据分别做一次验证集，其余的K-1组子集数据作为训练集，这样会得到K个模型，用这K个模型最终的验证集的分类准确率的平均数作为此K-CV下分类器的性能指标。
grid_model.fit(X_train,y_train)
grid_label=grid_model.predict(X_test)
print('准确率：',accuracy_score(pred_label,y_test))
print(grid_model.best_params_)
