## Web RecSys Project 

[![Build Status](https://travis-ci.org/HuGanghui/Web_RecSys_project.svg?branch=v1.0)](https://travis-ci.org/HuGanghui/Web_RecSys_project)
![PythonVersion](https://img.shields.io/badge/python-3.6-blue)
![PullRequest](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

Code framework ([中文版本](doc/Chinese-version.md)):
* **main.py** use the training set to search the grid to get the optimal parameters, 
    and then tests it on the test set, and outputs the test results in the result directory, 
    as well as the persistent prediction scores of each model

* **esamble_main.py** loads the prediction scores of multiple models to integrate the main function

* **parsejson/**

    **multi_parse.py** divides the original Yelp data set by city and supports multiple processes
    
    **topcity.py** statistically sorted by the city keyword
    
    **run.sh** runs the script for multi_parse

* **utility/**

    **split.py** the purpose of the class is to shard the data set and divide the review information of the same user proportionally
    The training set and test set, rather than the Spurise library of random partitioning, are a good way to solve the cold boot problem
    The partitioning ratio is controlled by test_size, and the filter_threshold is used to filter out only a small amount of review information
    User, the default user should have at least 2 review messages

    **util.py** some utility functions 

* **test/** contains all test functions

* **result/** model output

    **xxx_performance.txt** such a file as xxx_performance-txt contains the optimal parameters for the XXX model and rmse scores on the training and test sets

    **xxx_predict_result** is a persistent file for the prediction score of the test set by the XXX model, which can be reloaded to prepare for the subsequent esamble
    Persistence format and how to load reference test/test_load_result.py

* **config.ini** configures the data set path

* **multialgo.py** selects the parameters of the algorithm model, opens the multi-process model, and each algorithm model opens a process, and outputs the results of each algorithm model in the result directory

* **DeepLearningModel/** includes DeepcoNN model (https://github.com/chenchongthu/DeepCoNN) and [NARRE model] (https://github.com/chenchongthu/NARRE)
    The required tensorflow0.12.1 version of the dockerfile file
    It is convenient for everyone to build the environment, especially when cuda and cudnn versions of the server are too high to be compatible.

    **Dockerfile** builds the tensorflow0.12.1 version of the Dockerfile file

    **pip.conf**  pip domestic image configuration script

    **sources.list** Ubuntu16.04 domestic mirror configuration script

    **run.sh** runs the sample script for nvidia-docker

* **requirements.txt** dependent libraries