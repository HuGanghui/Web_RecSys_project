#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   sample_test.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2020/3/15 2:36 PM   hgh      1.0         None

import unittest


class SampleTest(unittest.TestCase):
    def testSample(self):
        self.assertEqual(4, 2 + 2)
