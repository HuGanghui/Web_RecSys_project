#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   split.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/14 3:36 PM   hgh      1.0         None
import configparser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Split:
    """
    该类的目的是来进行数据集切分，有目的的将同一个用户的review信息，按比例的分到
    训练集和测试集，而不是像Spurise库那样的随机划分，可以很好的解决冷启动的问题
    划分比例通过test_size来控制，filter_threshold用来过滤掉只有少量review信息的
    用户，默认用户至少要有2条review信息
    """
    def __init__(self, data, filter_threshold=1, test_size=0.2):
        self.data = data
        self.filter_threshold = filter_threshold
        self.test_size = test_size

    def _id_mapping(self, name):
        """
        mapping str in data to unique_int
        :param name: str, the name of column that need to be mapped

        """
        id_unique = np.unique(self.data[name])
        n = len(id_unique)
        dict_ = {}
        for i in np.arange(n):
            dict_[id_unique[i]] = i
        self.data[name] = self.data[name].map(dict_)

    def _user_filter(self, filter_threshold=1):
        """
        :param filter_threshold: int, the threshold of user

        """
        # 进行user过滤，要求同一个user至少有2条评论
        user_filter = self.data.groupby('user_id').size()
        user_filter = user_filter[user_filter > filter_threshold]
        self.data = self.data.set_index("user_id")
        self.data = self.data.loc[user_filter.index]
        self.data = self.data.reset_index()

    def _split(self, test_size=0.2):
        """
        split dataset that user_id as label, train_size = 0.8, test_size = 0.2
        :param test_size: int
        :return:
        train_set: pandas.DataFrame
        test_set: pandas.DataFrame
        """
        user_id_series = self.data["user_id"]
        # 切分数据集，按user_id的作为类别来切分，解决冷启动的问题
        X_train, X_test, _, _ = train_test_split(self.data, user_id_series, test_size=test_size,
                                                 stratify=user_id_series, random_state=11)
        return X_train, X_test

    def split(self):
        # 进行business_id到序号的mapping
        self._id_mapping("business_id")
        # 进行user_id到序号的mapping
        self._id_mapping("user_id")
        # 进行user过滤，要求同一个user至少有2条评论
        self._user_filter(self.filter_threshold)
        # # 切分数据集
        train_set, test_set = self._split(self.test_size)
        return train_set, test_set


if __name__ == '__main__':
    # example
    conf = configparser.ConfigParser()
    conf.read("../config.ini", encoding='utf-8')
    review_file_path = conf.get("path", "review_file_path")
    # review_file_path1 = conf.get("path", "review_file_path1")
    # review_file_path2 = conf.get("path", "review_file_path2")
    review_pd = pd.read_json(review_file_path, lines=True)
    print(review_pd.shape)
    review_pd = review_pd.loc[:, ["user_id", "business_id", "stars"]]
    split = Split(review_pd)
    train_set, test_set = split.split()
    pass
