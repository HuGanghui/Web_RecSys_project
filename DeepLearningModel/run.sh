#!/usr/bin/env bash
# 启动样例
# --name 是该docker进程的名字，不指定随机分配 -v 是将自己的目录进行挂载到docker中
nvidia-docker run -it --restart=always --name hghtf0.12 -v /data/HuGanghui/Pyproject/DeepCoNN:/home/hgh/DeepcoNN hgh/tensorflow:cuda8.0-cudnn5-devel-ubuntu16.04-tf0.12.1-py2.7 /bin/bash