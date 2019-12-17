#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   city.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/12/8 3:13 PM   hgh      1.0         None
import json


def city_sorted(file_path):
    """
    按city关键字，进行统计排序
    :param file_path:
    :return:
    """
    city_dict = {}
    with open(file_path) as f:
        for line in f:
            a = json.loads(line)
            if a["city"] in city_dict:
                city_dict[a["city"]] += 1
            else:
                city_dict[a["city"]] = 1
    sorted_city_dict = sorted(city_dict, key=city_dict.__getitem__, reverse=True)
    with open("./result_city_sorted.txt", "w") as out:
        for city in sorted_city_dict:
            out.write(city + " : " + str(city_dict[city]) + "\n")

