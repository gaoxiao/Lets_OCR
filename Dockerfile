FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER gaoxiao
LABEL version="1.0"
ENV LANG C.UTF-8
RUN apt-get update
RUN apt-get install libgtk2.0-dev libsm6 libxrender1 libxext-dev gcc python2.7 python-pip -y
##下载Anaconda3 python 环境安装包 放置在chineseocr目录 url地址https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
#RUN cd /chineseocr/text/detector/utils && sh make-for-cpu.sh
#RUN conda clean -p
#RUN conda clean -t
