#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   test_load_result.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/19 4:18 PM   hgh      1.0         None
import pickle


if __name__ == '__main__':
    # test
    result_path = "./test_data/test_SVD_predict_result"
    with open(result_path, "rb") as f:
        data = pickle.load(f)