## Web RecSys Project 

使用版本 python 3.6

代码框架：
* main 使用训练集进行网格搜索后获取最优的参数,然后在测试集上进行测试，并输出测试结果
* parsejson/

  multi_parse 将原始Yelp数据集按城市进行划分，支持多进程
  
  topcity 按city关键字，进行统计排序
  
  run.sh 运行multi_parse的脚本
  
* utility/
  
  split 该类的目的是来进行数据集切分，有目的的将同一个用户的review信息，按比例的分到
    训练集和测试集，而不是像Spurise库那样的随机划分，可以很好的解决冷启动的问题
    划分比例通过test_size来控制，filter_threshold用来过滤掉只有少量review信息的
    用户，默认用户至少要有2条review信息
  
* config.ini 配置数据集路径

* multialgo 进行算法模型参数选择，开启多进程模型，每个算法模型都开启一个进程，并在result目录下输出每个算法模型的结果

* requirements 需要安装的库