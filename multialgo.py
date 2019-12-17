#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   jsonpractice.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/8 3:18 PM   hgh      1.0         None
import multiprocessing
import pandas as pd
import numpy as np
from surprise import accuracy
from surprise.model_selection import cross_validate, KFold, GridSearchCV


class MultiAlgo(multiprocessing.Process):
    """
    进行算法模型参数选择，开启多进程模型，每个算法模型都开启一个进程，并在result目录下输出每个算法模型的结果
    """
    def __init__(self, processing_name, algo, param_grid, train_data, test_data, result_dir_path, name):
        multiprocessing.Process.__init__(self)
        self.algo = algo
        self.param_grid = param_grid
        self.data = train_data
        self.test_data = test_data
        self.result_dir_path = result_dir_path
        self.name = name
        self.processing_name = processing_name
        self.best_model = None

    def single_algo_test(self):
        gs = GridSearchCV(self.algo, self.param_grid, measures=['rmse', 'mae'], cv=5)
        print(str(self.algo))
        gs.fit(self.data)
        with open(self.result_dir_path + "/" + self.name + "_performace.txt", "w") as f:
            # best RMSE score
            f.write("best_rmse: %f" % gs.best_score['rmse'] + "\n")
            # combination of parameters that gave the best RMSE score
            f.write("best_params of rmse: " + str(gs.best_params['rmse']) + "\n")
        self.best_model = gs.best_estimator["rmse"]
        self._best_model_predict()

    def _best_model_predict(self):
        self.best_model.fit(self.data.build_full_trainset())
        raw_testset = [self.test_data.raw_ratings[i] for i in np.arange(len(self.test_data.raw_ratings))]
        test_data = self.test_data.construct_testset(raw_testset)
        pred = self.best_model.test(test_data)
        test_true = np.asarray([pred[i][2] for i in range(len(pred))])
        test_pred = np.asarray([pred[i][3] for i in range(len(pred))])
        with open(self.result_dir_path + "/" + self.name + "_performace.txt", "a") as f:
            f.write("test_set's rmse :" + str(np.sqrt(np.mean(np.square(test_true-test_pred)))))
        accuracy.rmse(pred)

    def run(self):
        print("开始进程：" + self.processing_name)
        self.single_algo_test()
        print("退出进程：" + self.processing_name)


if __name__ == '__main__':
    pass
