#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   util.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/19 5:31 PM   hgh      1.0         None
import numpy as np


def rmse(true, pred):
    """
    :param true: numpy.darray, 真实评分
    :param pred: numpy.darray, 预测评分
    :return:
    rmse float rmse分数
    """
    return np.sqrt(np.mean(np.square(true - pred)))

