#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   test_multialgo.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/30 7:23 PM   hgh      1.0         None
from surprise import SVD

from multialgo import MultiAlgo

if __name__ == '__main__':
    # 测试
    # test_SVD_param_grid = {"n_factors": [3], "n_epochs": [30], "biased": [True],
    #                        "lr_all": [0.002], "reg_all": [0.02]}
    # test_name = "test_svd"
    #
    # test_pure_SVD_param_grid = {"n_factors": [3], "n_epochs": [30], "biased": [False],
    #                        "lr_all": [0.002], "reg_all": [0.02]}
    # test_name1 = "test_pure_svd"
    # test_model = SVD
    # process = MultiAlgo("Provessing-" + str(1), test_model, test_pure_SVD_param_grid, train_data, test_data, result_dir_path, test_name1)
    # process.start()
    pass