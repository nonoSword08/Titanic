#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Survived:0代表死亡，1代表存活
Pclass:乘客所持票类，有三种值(1,2,3)
Name:乘客姓名
Sex:乘客性别
Age:乘客年龄(有缺失)
SibSp:乘客兄弟姐妹/配偶的个数(整数值)
Parch:乘客父母/孩子的个数(整数值)
Ticket:票号(字符串)
Fare:乘客所持票的价格(浮点数，0-500不等)
Cabin:乘客所在船舱(有缺失)
Embarked:乘客登船的地点
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_survived_rate(df):
    """
    计算存活率
    :param df: data_frame对象
    :return: 存活率
    """
    survived_rate = float(df['Survived'].sum()) / df['Survived'].count()
    return survived_rate


def survived_rate_with_pclass(df):
    """
    得到存活率与票类的关系
    :param df: data_frame对象
    :return: 存活率
    """
    # x = [df[(df.Pclass == 1)]['Pclass'].size,
    #      df[(df.Pclass == 2)]['Pclass'].size,
    #      df[(df.Pclass == 3)]['Pclass'].size,
    #      ]
    # y = [df[(df.Pclass == 1) & (df.Survived == 1)]['Pclass'].size,
    #      df[(df.Pclass == 2) & (df.Survived == 1)]['Pclass'].size,
    #      df[(df.Pclass == 3) & (df.Survived == 1)]['Pclass'].size,
    #      ]

    pclass_survived_rate = (df.groupby(['Pclass']).sum() / df.groupby(['Pclass']).count())['Survived']
    pclass_survived_rate.plot(kind='pie')
    plt.title('survived_rate_with_pclass')
    plt.show()


def survived_rate_with_sex(df):
    """
    得到存活率与性别的关系
    :param df:
    :return:
    """
    print(df.groupby(['Sex']).sum())
    print(df.groupby(['Sex']).count())
    sex_survived_rate = (df.groupby(['Sex']).sum() / df.groupby(['Sex']).count())['Survived']
    sex_survived_rate.plot(kind='bar')
    plt.title('survived_rate_with_sex')
    plt.show()


def survived_rate_with_age(df):
    """
    得到存活率与年龄的关系
    :param df:
    :return:
    """
    age_clean_data = df[~np.isnan(df['Age'])]
    ages = np.arange(0, 81, 5)
    age_cut = pd.cut(age_clean_data.Age, ages)
    age_cut_grouped = age_clean_data.groupby(age_cut)
    age_survived_rate = (age_cut_grouped.sum() / age_cut_grouped.count())['Survived']
    age_survived_rate.plot(kind='bar')
    plt.title('survived_rate_with_age')
    plt.show()


def survived_rate_with_sex_and_pclass(df):
    sex_and_class_survived_rate = (df.groupby(['Sex', 'Pclass']).sum() / df.groupby(['Sex', 'Pclass']).count())['Survived']
    sex_and_class_survived_rate.plot(kind='bar')
    plt.title('survived_rate_with_sex_and_pclass')
    plt.show()

if __name__ == '__main__':
    data_src = './titanic_train.csv'
    data_frame = pd.read_csv(data_src, header=0)
    # # 获取数据文件信息
    # print(data_frame.info())
    # # 获取数据的摘要
    # print(df.describe())
    # # 看看前几行数据
    # print(df.head())

    print('存活率为:%.3f' % get_survived_rate(data_frame))

    # # 存活率与票类的关系
    # survived_rate_with_pclass(data_frame)
    # # 存活率和性别的关系
    # survived_rate_with_sex(data_frame)
    # 存活率与年龄的关系
    survived_rate_with_age(data_frame)
