# 创建 docker build -t hgh/tensorflow:cuda8.0-cudnn5-devel-ubuntu16.04-tf0.12.1-py2.7
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV PATH /opt/miniconda2/bin:$PATH

# 更改ubuntu16.04源镜像
RUN cd /etc/apt/ \
    && mv sources.list sources.list.bak

COPY sources.list /etc/apt/

# 配置pip源镜像
COPY pip.conf /root/.pip/
COPY pip.conf /home/hgh/.pip/

RUN buildDeps='wget openssh-server net-tools git' \
    && apt-get update \
    && apt-get install -y $buildDeps \
    && rm -rf /var/lib/apt/lists/* \
    # 卸载自带的编辑器，安装vim
    && apt-get update \
    && apt-get remove vim-common \
    && apt-get install -y vim

# 配置ssh
RUN mkdir -p /var/run/sshd \
    && echo PermitRootLogin yes >> /etc/ssh/sshd_config \
    && echo root:123456 | chpasswd

RUN  wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda2-4.7.12.1-Linux-x86_64.sh -O ~/miniconda2.sh \
    && /bin/bash ~/miniconda2.sh -b -p /opt/miniconda2 \
    && rm ~/miniconda2.sh \
    && echo "export PATH=/opt/miniconda2/bin:$PATH" >> ~/.bashrc

RUN pip install --no-cache-dir numpy==1.11.1 \
    && pip install --no-cache-dir pandas==0.21.0 \
    && pip install --no-cache-dir tensorflow-gpu==0.12.1

# hgh 为用户名可相应更改为自己的
RUN useradd --create-home --no-log-init --shell /bin/bash hgh \
&& adduser hgh sudo \
&& echo 'hgh:123456' | chpasswd \
&& chown -R hgh /home/hgh

USER hgh
WORKDIR /home/hgh

EXPOSE 22 8888 6006

