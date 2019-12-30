#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   esamble_main.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/19 4:58 PM   hgh      1.0         None
import pickle

from sklearn.linear_model import LinearRegression, Ridge

from utility.util import rmse
import pandas as pd
import numpy as np


def get_pred(result_path_list, name_list, label):
    pred_list = []
    for i in np.arange(len(result_path_list)):
        with open(result_path_list[i], "rb") as f:
            data = pickle.load(f)
            pred = data["pred"]
            print(name_list[i] + " rsme : " + str(rmse(label, pred)))
            pred_list.append(pred)
    return pd.DataFrame(pred_list).T


if __name__ == '__main__':
    # result_path = "./result/test_SVD_predict_result"
    # with open(result_path, "rb") as f:
    #     svd_data = pickle.load(f)
    #     svd_pred = svd_data["pred"]
    # result_path1 = "./result/test_pure_SVD_predict_result"
    # with open(result_path1, "rb") as f:
    #     pure_svd_data = pickle.load(f)
    #     pure_svd_pred = pure_svd_data["pred"]

    train_path = "./result/train_label"
    with open(train_path, "rb") as f:
        train_label = pickle.load(f)

    test_path = "./result/test_label"
    with open(test_path, "rb") as f:
        test_label = pickle.load(f)

    # 目前使用简单对指定系数，后续思考方法
    # esamble_pred = 0.5 * svd_pred + 0.5 * pure_svd_pred
    # print("svd rsme : " + str(rmse(test_label, svd_pred)))
    # print("pure svd rsme : " + str(rmse(test_label, pure_svd_pred)))
    # print("esamble rsme : " + str(rmse(test_label, esamble_pred)))

    test_path_list = ["./result/biased_SVD_predict_result",
                      "./result/SVDpp_predict_result",
                      "./result/biased_NMF_predict_result",
                      ]

    train_path_list = ["./result/biased_SVD_train_predict_result",
                       "./result/SVDpp_train_predict_result",
                       "./result/biased_NMF_train_predict_result",
                       ]
    name_list = ["biased_SVD",
                 "SVDpp",
                 "biased_NMF",
                 ]
    train_pred_pd = get_pred(train_path_list, name_list, train_label)
    test_pred_pd = get_pred(test_path_list, name_list, test_label)

    # linear_model = LinearRegression(fit_intercept=False)
    linear_model = Ridge(alpha=40000, solver="sag", fit_intercept=False, max_iter=10000)
    linear_model.fit(train_pred_pd, train_label)
    print(str(linear_model.coef_))
    y_train_pred = linear_model.predict(train_pred_pd)
    y_test_pred = linear_model.predict(test_pred_pd)
    print("esamble result(train): " + str(rmse(train_label, y_train_pred)))
    print("esamble result(test): " + str(rmse(test_label, y_test_pred)))
    pass
