#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   test.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/3 9:10 PM   hgh      1.0         None
import json
import multiprocessing
import time
import pandas as pd
import configparser
import sys


class SampleCityFromJsonData(multiprocessing.Process):
    """
    将原始数据按城市进行划分处理，提供多进程
    """
    def __init__(self, processing_name, in_dir_path, out_dir_path, city_name):
        multiprocessing.Process.__init__(self)
        self.processing_name = processing_name
        self.in_dir_path = in_dir_path
        self.out_dir_path = out_dir_path
        self.city_name = city_name

    def _sample_from_business(self, in_file_path, out_file_path):
        # BUSINESS.JSON
        business_data = []
        print('Reading business.json')
        # TODO 这里可以之间pd.read_json
        with open(in_file_path) as f:
            for line in f:
                business_data.append(json.loads(line))
        business_df = pd.DataFrame.from_dict(business_data)
        business_sample = business_df[business_df['city'].str.contains(self.city_name)]
        print('Creating business_sample.json')
        with open(out_file_path, 'w') as f:
            business_sample.to_json(f, orient='records', lines=True)
        # now that sample file is made, get list of business id's from it
        # convert into a dict with values as keys for fast searching
        print('Getting business_ids')
        self.business_ids = pd.Series(business_sample['business_id'].index.values,
                                      index=business_sample['business_id']).to_dict()

    def _sample_from_review(self, in_file_path, out_file_path):
        # REVIEW.JSON
        review_data = []
        print('Reading review.json')
        with open(in_file_path) as f:
            for line in f:
                newline = eval(line)  # read the line (str) as a dict
                # TODO 这里是不是可以做并行操作，然后进行合并，其他是类似的
                if newline['business_id'] in self.business_ids:
                    review_data.append(json.loads(line))
        review_df = pd.DataFrame.from_dict(review_data)
        # get list of business id's from business.json and only keep reviews of those businesses
        # would normally write it with to_json but it's too big to fit in memory, so write it line by line
        print('Creating review_sample.json line by line')
        for row in review_df.iterrows():
            with open(out_file_path, 'a') as f:
                line = row[1].to_json(f)
                f.write('\n')
        # now that sample file is made, get list of user id's from it
        # convert into a dict with values as keys for fast searching
        print('Getting user_ids')
        self.user_ids = pd.Series(review_df['user_id'].index.values, index=review_df['user_id']).to_dict()

    def _sample_from_user(self, in_file_path, out_file_path):
        # USER.JSON
        user_data = []
        with open(in_file_path) as f:
            for line in f:
                newline = eval(line)  # read the line (str) as a dict
                if newline['user_id'] in self.user_ids:
                    user_data.append(json.loads(line))
        user_df = pd.DataFrame.from_dict(user_data)
        print('Creating user_sample.json')
        with open(out_file_path, 'w') as f:
            user_df.to_json(f, orient='records', lines=True)

    def _sample_from_tip(self, in_file_path, out_file_path):
        # TIP.JSON
        tip_data = []
        print('Reading tip.json')
        with open(in_file_path) as f:
            for line in f:
                newline = eval(line)
                if newline['business_id'] in self.business_ids:
                    tip_data.append(json.loads(line))
        tip_df = pd.DataFrame.from_dict(tip_data)
        print('Creating tip_sample.json')
        with open(out_file_path, 'w') as f:
            tip_df.to_json(f, orient='records', lines=True)

    def _sample_from_checkin(self, in_file_path, out_file_path):
        # CHECKIN.JSON
        checkin_data = []
        print('Reading checkin.json')
        with open(in_file_path) as f:
            for line in f:
                newline = eval(line)
                if newline['business_id'] in self.business_ids:
                    checkin_data.append(json.loads(line))
        checkin_df = pd.DataFrame.from_dict(checkin_data)
        print('Creating checkin_sample.json')
        with open(out_file_path, 'w') as f:
            checkin_df.to_json(f, orient='records', lines=True)

    def sample(self):
        # input file
        business_file = self.in_dir_path + "/yelp_academic_dataset_business.json"
        review_file = self.in_dir_path + "/yelp_academic_dataset_review.json"
        user_file = self.in_dir_path + "/yelp_academic_dataset_user.json"
        tip_file = self.in_dir_path + "/yelp_academic_dataset_tip.json"
        checkin_file = self.in_dir_path + "/yelp_academic_dataset_checkin.json"
        # output file
        business_sample_file = self.out_dir_path + "/" + self.city_name + "_business.json"
        review_sample_file = self.out_dir_path + "/" + self.city_name + "_review.json"
        user_sample_file = self.out_dir_path + "/" + self.city_name + "_user.json"
        tip_sample_file = self.out_dir_path + "/" + self.city_name + "_tip.json"
        checkin_sample_file = self.out_dir_path + "/" + self.city_name + "_checkin.json"
        # start sample
        self._sample_from_business(business_file, business_sample_file)
        self._sample_from_review(review_file, review_sample_file)
        self._sample_from_user(user_file, user_sample_file)
        self._sample_from_tip(tip_file, tip_sample_file)
        self._sample_from_checkin(checkin_file, checkin_sample_file)

    def run(self):
        print("开始进程：" + self.processing_name)
        self.sample()
        print("退出进程：" + self.processing_name)


if __name__ == '__main__':
    # test city_sorted
    # dir = os.path.dirname(os.path.dirname(__file__))
    # business_file = dir + "/yelp_academic_dataset_business.json"
    # city_sorted(business_file)

    # test msample_from_business
    conf = configparser.ConfigParser()
    conf.read("../config.ini", encoding='utf-8')
    in_dir_path = conf.get("path", "in_dir_path")
    out_dir_path = conf.get("path", "out_dir_path")
    for idx, city_name in enumerate(sys.argv[1:]):
        # 创建新进程
        process = SampleCityFromJsonData("Processing-" + str(idx), in_dir_path, out_dir_path, city_name)
        process.start()
