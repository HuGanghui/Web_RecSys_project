#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   test.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/12 11:06 PM   hgh      1.0         None
import configparser

import pandas as pd
from surprise import SVD, Reader, Dataset, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, CoClustering
import numpy as np


from multialgo import MultiAlgo
from utility.split import Split


if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read("./config.ini", encoding='utf-8')
    review_file_path = conf.get("path", "review_file_path")
    review_file_path1 = conf.get("path", "review_file_path1")
    # review_file_path2 = conf.get("path", "review_file_path2")
    result_dir_path = conf.get("path", "result_dir_path")
    review_pd = pd.read_json(review_file_path1, lines=True)

    print(review_pd.shape)
    review_pd = review_pd.loc[:, ["user_id", "business_id", "stars"]]
    split = Split(review_pd)
    train_set, test_set = split.split()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(review_pd, reader)
    test_data = Dataset.load_from_df(test_set, reader)
    classes = (SVD, SVD, SVDpp, NMF, NMF, KNNBasic, KNNWithMeans, KNNBaseline,
               KNNWithZScore, CoClustering)
    names = ("pure_SVD", "biased_SVD", "SVDpp", "pure_NMF", "biased_NMF", "KNNBasic", "KNNWithMeans",
             "KNNBaseline", "KNNWithZScore", "CoClustering")
    test_SVD_param_grid = {"n_factors": [3], "n_epochs": [30], "biased": [True],
                           "lr_all": [0.002], "reg_all": [0.02]}
    pure_SVD_param_grid = {"n_factors": [3, 5, 10, 15], "n_epochs": [10, 20, 30], "biased": [False],
                           "lr_all": [0.002, 0.005], "reg_all": [0.02, 0.04, 0.06]}
    biased_SVD_param_grid = {"n_factors": [3, 5, 10, 15], "n_epochs": [10, 20, 30], "biased": [True],
                             "lr_all": [0.002, 0.005], "reg_all": [0.02, 0.04, 0.06]}
    SVDpp_param_grid = {"n_factors": [3, 5, 10], "n_epochs": [10, 20],
                        "lr_all": [0.002, 0.005], "reg_all": [0.02, 0.04]}
    pure_NMF_param_grid = {"n_factors": [3, 5, 10], "n_epochs": [10, 20], "biased": [False],
                           "reg_pu": [0.04, 0.06], "reg_qi": [0.04, 0.06]}
    biased_NMF_param_grid = {"n_factors": [3, 5, 10], "n_epochs": [10, 20], "biased": [True],
                             "reg_pu": [0.04, 0.06], "reg_qi": [0.04, 0.06], "reg_bu": [0.02, 0.04],
                             "reg_bi": [0.02, 0.04], "lr_bu": [0.004, 0.005],
                             "lr_bi": [0.004, 0.005]}
    KNN_param_grid = {"k": [10], "sim_options": {'name': ["msd"],
                                                            "user_based": [True]}}
    CoClustering_param_grid = {"n_cltr_u": [3, 5, 7, 10], "n_cltr_i": [3, 5, 7, 10], "n_epochs": [10, 20, 30]}
    param_grid = (pure_SVD_param_grid, biased_SVD_param_grid, SVDpp_param_grid, pure_NMF_param_grid,
                  biased_NMF_param_grid, KNN_param_grid, KNN_param_grid, KNN_param_grid, KNN_param_grid,
                  CoClustering_param_grid)

    # 使用训练集进行网格搜索后获取最优的参数
    for i in np.arange(len(classes)):
        process = MultiAlgo("Processing-" + str(i), classes[i], param_grid[i], data, test_data, result_dir_path, names[i])
        process.start()

    # 单个进行测试
    # process = MultiAlgo("Provessing-" + str(1), classes[1], test_SVD_param_grid, data, test_data, result_dir_path, "test_SVD")
    # process.start()






