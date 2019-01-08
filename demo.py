#-*- coding:utf-8 -*-
"""
@Time      :2018/9/23 11:46
@Author    :Sunkai
@Email     :892752842@qq.com
@File      :机器学习/逻辑回归
@Software  :Pycharm Community Edition
"""
import pandas as pd
import numpy as np

#读取训练集
train = pd.read_csv("train.csv")
#读取训练集
test = pd.read_csv("test.csv")
# print("训练集：",train.shape,"\n测试集：",test.shape)

#合并数据集，便于对训练集和预测集清洗处理
full = train.append(test,ignore_index=True)
# print("合并数据集：",full.shape)

#描述性统计
# print(full.describe())#只能对数值型数据统计
# print(full.info())  #对异常字符串判断

#数据类型缺失值补充（fillna）
#年龄
full['Age'] = full['Age'].fillna(full['Age'].mean())#平均值填充
#船票价格
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())

#字符串类型缺失值处理
#登船巷口（Embarked）
'''
出发地点：S=英国 南安普顿
途径地点1：C=法国 瑟堡市
途径地点2：Q=爱尔兰 昆士敦
'''
#观察数据Embarked只有两个缺失值，我们将最频繁出现的S：英国南安普顿填充
full['Embarked'] = full['Embarked'].fillna('S')
#船舱号缺失数据较多，用U填充表示未知
full['Cabin'] = full['Cabin'].fillna('U')

#特征工程  one_hot编码
#分类数据提取：性别
'''
男对应为1，女对应为0
'''
sex_mapDict = {'male':1,'female':0}
full['Sex'] = full['Sex'].map(sex_mapDict)
sexDF = pd.DataFrame()
sexDF = pd.get_dummies(full['Sex'])
# print(sexDF.head())

#登陆港口
#查看原数据
# print(full['Embarked'].head())
#存放提取后的特征
embaredDf = pd.DataFrame()
#使用get_dummis进行one hot编码，列名前缀为Embared
embaredDf = pd.get_dummies(full['Embarked'],prefix='Embarked')
print(embaredDf)

#客舱one_hot编码
PclassDF = pd.DataFrame()
PclassDF = pd.get_dummies(full['Pclass'],prefix='Pclass')
# print(PclassDF.head())

#分类数据特征提取：姓名
'''
定义函数，从名字中获取头衔
'''
def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()  #用于一处字符串头尾指定的字符
    return str3

titleDF = pd.DataFrame()
titleDF['Title'] = full['Name'].map(getTitle)
# print(titleDF.head())

#姓名头衔映射关系
title_mapDict = {
            "Capt"   : "Officer",
            "Col"    : "Officer",
            "Major"  :"Officer",
            "Don"    : "Royalty",
            "Sir"    : "Royalty",
            "Dr"     : "Officer",
            "Rev"    :"Officer",
    "the Countess"   :"Royalty",
    "Dona"           :"Royalty",
    "Mme"            :"Mrs",
    "Mlle"           :"Miss",
    "Ms"             :"Mrs",
    "Mr"             :"Mr",
    "Mrs"            :"Mrs",
    "Miss"           : "Miss",
    "Master"         :"Master",
    "Lady"           :"Royalty",
}
titleDF['Title'] = titleDF['Title'].map(title_mapDict)
#使用get_dummis进行one hot编码
titleDF = pd.get_dummies(titleDF['Title'])
# print(titleDF.head())

#分类数据特征提取：客舱号
'''
匿名函数用法
lambda 参数1，参数2：函数体
'''
#定义函数体
CabinDF = pd.DataFrame()
'''
客场号的类别值是首字母，例如
C85 类别映射为首字母C
'''
full['Cabin'] = full['Cabin'].map(lambda c:c[0])
# print(full['Cabin']) #将客场号的首字母提取出来
#使用get_dummis进行one hot编码，列名前缀为Cabin
cabinDF = pd.get_dummies(full['Cabin'],prefix='Cabin')
# print(cabinDF.head())

#分类数据特征的提取：家庭类别
#存放家庭信息
familyDf = pd.DataFrame()
'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
'''
familyDf['FamilySize'] = full['Parch']+full['SibSp']+1
# print(familyDf) #家庭人数
'''
家庭类别：
小家庭（Family_Single）：家庭人数=1
中家庭（Family_Small）:2<=家庭人数<=4
大家庭（Family_Large）:家庭人数>=5
'''
#if条件为真则返回if前面的值，否则返回0
familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s:1 if s==1 else 0 )
familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s:1 if 2<=s<=4 else 0 )
familyDf['Family_Large'] = familyDf['FamilySize'].map(lambda s:1 if s>=5 else 0 )
# print(familyDf.head())

#特征选择和特征降维
#数据的特征选择：相关系数
corrDf = full.corr()
# print(corrDf)

'''
查看各个特征与生存情况（Survived）的相关系数
ascending=False表示按照降序排列
'''
# print(corrDf['Survived'].sort_values(ascending=False))
#特征选择,Concat方法把这些值按照列合成一个新的数据框
full_X = pd.concat([
    titleDF,#头衔
    PclassDF,#客舱等级
    familyDf,#家庭大小
    full['Fare'],#船票价格
    cabinDF,#船舱号
    embaredDf,#登船巷口
    full['Sex'],#性别
    # sexDF,
    full['Age'],
],axis=1)
# print(full_X.head())

#构建模型
#原始数据共有891行
sourceRow = 891
#原始数据集，特征
source_X = full_X.loc[0:sourceRow-1,:]
print(source_X)
# 原始数据集，标签
source_y = full.loc[0:sourceRow-1,'Survived']
#预测数据集，特征
pred_X = full_X.loc[sourceRow:,:]

#训练数据和测试数据
from sklearn.model_selection import train_test_split
#建立模型用的训练数据集和测试数据集
train_X,test_X,train_y,test_y = train_test_split(source_X,source_y,random_state = 20,train_size=.8)
# print(train_X)
# print(test_X)
#输出数据集大小
# print('原始数据特征:',  source_X.shape,
#       '\n训练数据特征:',train_X.shape,
#       '\n测试数据特征:',test_X.shape)
# print('原始数据标签:',  source_y.shape,
#       '\n训练数据标签:',train_y.shape,
#       '\n测试数据标签:',test_y.shape)

#训练模型
#第一步：导入算法
from sklearn.linear_model import LogisticRegression
#第二步：创建逻辑回归模型
model = LogisticRegression()
#第三步：训练模型
model.fit(train_X,train_y)#train_X为训练数据特征，train_y为训练数据标签

#分类问题，score得到的是模型的准确率
print(model.score(test_X,test_y))

#方案实施，预测
pred_Y = model.predict(pred_X)
'''
生成的预测值为浮点数（0.0,1,0）
若要求为整形，则需要转化数据类型
'''
pred_Y = pred_Y.astype(int)
#乘客id
passenger_id = full.loc[sourceRow:,'PassengerId']
#数据框，乘客ID，预测生存情况的值
preDf = pd.DataFrame({
    'Passenger_Id':passenger_id,
    'Survived':pred_Y
})
# print(preDf.shape)
# preDf.to_csv('titanic_pred.csv',index=False)