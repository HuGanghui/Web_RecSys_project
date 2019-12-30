#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   esamble_main.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/19 4:58 PM   hgh      1.0         None
import pickle

from utility.util import rmse


def get_pred(result_path):
    with open(result_path, "rb") as f:
        data = pickle.load(f)
        pred = data["pred"]
    return pred


if __name__ == '__main__':
    result_path = "./result/test_SVD_predict_result"
    with open(result_path, "rb") as f:
        svd_data = pickle.load(f)
        svd_pred = svd_data["pred"]
    result_path1 = "./result/test_pure_SVD_predict_result"
    with open(result_path1, "rb") as f:
        pure_svd_data = pickle.load(f)
        pure_svd_pred = pure_svd_data["pred"]
    test_path = "./result/test_label"
    with open(test_path, "rb") as f:
        test_label = pickle.load(f)
    path_list = ["result/biased_SVD_predict_result",
                 "./result/SVDpp_predict_result"]
    name_list = ["biased_SVD", "SVDpp"]
    # esamble_pred = 0
    # for i in range(len(path_list)):
    #     pred = get_pred(path_list[i])
    #     esamble_pred += 1 / 2 * pred
    #     print(name_list[i] + " rsme : " + str(rmse(test_label, pred)))

    # 目前使用简单对指定系数，后续思考方法
    esamble_pred = 0.5 * svd_pred + 0.5 * pure_svd_pred
    print("svd rsme : " + str(rmse(test_label, svd_pred)))
    print("pure svd rsme : " + str(rmse(test_label, pure_svd_pred)))
    print("esamble rsme : " + str(rmse(test_label, esamble_pred)))
    pass