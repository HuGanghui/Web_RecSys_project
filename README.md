## Web RecSys Project 

[![Build Status](https://travis-ci.org/HuGanghui/Web_RecSys_project.svg?branch=v1.0)](https://travis-ci.org/HuGanghui/Web_RecSys_project)
![PythonVersion](https://img.shields.io/badge/python-3.6-blue)
![PullRequest](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

代码框架：
* main.py 使用训练集进行网格搜索后获取最优的参数,然后在测试集上进行测试，并在result目录输出测试结果，以及每个模型的持久化的预测分数
* esamble_main.py 加载多个模型的预测分数，进行集成的主函数
* parsejson/

  multi_parse.py 将原始Yelp数据集按城市进行划分，支持多进程
  
  topcity 按city关键字，进行统计排序
  
  run.sh 运行multi_parse的脚本
  
* utility/
  
  split.py 该类的目的是来进行数据集切分，有目的的将同一个用户的review信息，按比例的分到
    训练集和测试集，而不是像Spurise库那样的随机划分，可以很好的解决冷启动的问题
    划分比例通过test_size来控制，filter_threshold用来过滤掉只有少量review信息的
    用户，默认用户至少要有2条review信息
  
  util.py 一些效用函数
  
* test/ 包含所有测试函数

* result/ 模型输出结果

  xxx_performance.txt 此类文件是包含xxx模型的最优参数和在训练集以及测试集上的rmse分数
  
  xxx_predict_result 此类文件为xxx模型对测试集预测评分的持久化文件，可重新加载，为后续esamble做准备
  持久化格式和如何加载可参考 test/test_load_result.py
  
* config.ini 配置数据集路径

* multialgo.py 进行算法模型参数选择，开启多进程模型，每个算法模型都开启一个进程，并在result目录下输出每个算法模型的结果

* DeepLearningModel/ 包含了[DeepcoNN模型](https://github.com/chenchongthu/DeepCoNN)
    和 [NARRE模型](https://github.com/chenchongthu/NARRE) 
    所需要的tensorflow0.12.1版本的dockerfile文件
    方便大家构建环境，尤其是当服务器的cuda和cudnn版本太高，无法兼容时。
    
    Dockerfile 构建tensorflow0.12.1版本的dockerfile文件
    
    pip.conf pip国内镜像配置脚本
    
    sources.list Ubuntu16.04国内镜像配置脚本
    
    run.sh 运行nvidia-docker的脚本样例

* requirements 需要安装的库
