#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   test_multialgo.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/30 7:23 PM   hgh      1.0         None
import configparser
import sys

from surprise import SVD, Reader, Dataset
import pandas as pd
from multialgo import MultiAlgo
from utility.split import Split

if __name__ == '__main__':
    # 测试
    # 获取数据路径
    conf = configparser.ConfigParser()
    conf.read("../config.ini", encoding='utf-8')
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
    review_pd = pd.read_json("." + review_file_path1, lines=True)
    print(review_pd.shape)
    # 获取数据
    review_pd = review_pd.loc[:, ["user_id", "business_id", "stars"]]
    # 进行切分
    split = Split(review_pd)
    train_set, test_set = split.split()
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_set, reader)
    test_data = Dataset.load_from_df(test_set, reader)

    test_SVD_param_grid = {"n_factors": [3], "n_epochs": [30], "biased": [True],
                           "lr_all": [0.002], "reg_all": [0.02]}
    test_name = "test_svd"

    test_pure_SVD_param_grid = {"n_factors": [3], "n_epochs": [30], "biased": [False],
                                "lr_all": [0.002], "reg_all": [0.02]}
    test_name1 = "test_pure_svd"
    test_model = SVD
    process = MultiAlgo("Provessing-" + str(1), test_model, test_pure_SVD_param_grid, train_data, test_data, result_dir_path, test_name1)
    process.start()
    pass