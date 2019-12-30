#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   test.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/12 11:06 PM   hgh      1.0         None
import configparser
import os
import pickle

import pandas as pd
from surprise import SVD, Reader, Dataset, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, CoClustering
import numpy as np

import sys
from multialgo import MultiAlgo
from utility.split import Split


if __name__ == '__main__':
    # 获取数据路径
    conf = configparser.ConfigParser()
    conf.read("./config.ini", encoding='utf-8')
    # 服务器是linux系统
    if str(sys.platform) == "linux":
        review_file_path = conf.get("linux_path", "review_file_path")
        review_file_path1 = conf.get("linux_path", "review_file_path1")
        # review_file_path2 = conf.get("linux_path", "review_file_path2")
        result_dir_path = conf.get("linux_path", "result_dir_path")
    else:
        review_file_path = conf.get("darwin_path", "review_file_path")
        review_file_path1 = conf.get("darwin_path", "review_file_path1")
        # review_file_path2 = conf.get("darwin_path", "review_file_path2")
        result_dir_path = conf.get("darwin_path", "result_dir_path")
    review_pd = pd.read_json(review_file_path1, lines=True)
    print(review_pd.shape)
    # 获取数据
    review_pd = review_pd.loc[:, ["user_id", "business_id", "stars"]]
    # 进行切分
    split = Split(review_pd)
    train_set, test_set = split.split()
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_set, reader)
    test_data = Dataset.load_from_df(test_set, reader)
    # 对train_set, test_set 标签进行持久化，为后续esamble准备
    test_label_file = result_dir_path + "/" + "test_label"
    if not os.path.exists(test_label_file):
        with open(test_label_file, 'wb') as f:
            pickle.dump(test_set["stars"].values, f)
    test_label = test_set["stars"].values
    train_label_file = result_dir_path + "/" + "train_label"
    if not os.path.exists(train_label_file):
        with open(train_label_file, 'wb') as f:
            pickle.dump(train_set["stars"].values, f)
    # 算法模型，并进行参数选择
    classes = (SVD, SVD, SVDpp, NMF, NMF, KNNBasic, KNNWithMeans, KNNBaseline,
               KNNWithZScore, CoClustering)
    names = ("pure_SVD", "biased_SVD", "SVDpp", "pure_NMF", "biased_NMF", "KNNBasic", "KNNWithMeans",
             "KNNBaseline", "KNNWithZScore", "CoClustering")
    pure_SVD_param_grid = {"n_factors": [3, 5, 10, 15], "n_epochs": [10, 20, 30], "biased": [False],
                           "lr_all": [0.002, 0.005], "reg_all": [0.02, 0.04, 0.06]}
    # 这已经是选出来的最好的参数了，目前是因为需求再跑一些，就不选择了，下面只有一组参数的同理
    biased_SVD_param_grid = {"n_factors": [3], "n_epochs": [30], "biased": [True],
                             "lr_all": [0.005], "reg_all": [0.06]}
    SVDpp_param_grid = {"n_factors": [3], "n_epochs": [20],
                        "lr_all": [0.005], "reg_all": [0.04]}
    pure_NMF_param_grid = {"n_factors": [3, 5, 10], "n_epochs": [10, 20], "biased": [False],
                           "reg_pu": [0.04, 0.06], "reg_qi": [0.04, 0.06]}
    biased_NMF_param_grid = {"n_factors": [3], "n_epochs": [10], "biased": [True],
                             "reg_pu": [0.06], "reg_qi": [0.06], "reg_bu": [0.04],
                             "reg_bi": [0.02], "lr_bu": [0.005],
                             "lr_bi": [0.005]}
    KNN_param_grid = {"k": [20], "sim_options": {'name': ["msd"], "user_based": [True]}}
    CoClustering_param_grid = {"n_cltr_u": [3, 5, 7, 10], "n_cltr_i": [3, 5, 7, 10], "n_epochs": [10, 20, 30]}
    param_grid = (pure_SVD_param_grid, biased_SVD_param_grid, SVDpp_param_grid, pure_NMF_param_grid,
                  biased_NMF_param_grid, KNN_param_grid, KNN_param_grid, KNN_param_grid, KNN_param_grid,
                  CoClustering_param_grid)

    # 使用训练集进行网格搜索后获取最优的参数,并用最优参数在测试集上测试
    # for i in [1, 2, 4]:
    #     process = MultiAlgo("Processing-" + str(i), classes[i], param_grid[i], train_data, test_data, result_dir_path, names[i])
    #     process.start()







