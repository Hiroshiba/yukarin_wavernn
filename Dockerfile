FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
SHELL ["/bin/bash", "-c"]

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y swig libsndfile1-dev libasound2-dev && \
    apt-get clean

WORKDIR /app

# pypi
RUN pip install llvmlite cython --ignore-installed

# cpp
# COPY src_cython /app/src_cython

# RUN git clone https://github.com/Hiroshiba/yukarin_autoreg_cpp && \
#     cd yukarin_autoreg_cpp && \
#     git checkout 094b63a36057c3a6385c51e702c672e3da2321b1 && \
#     git clone https://github.com/NVIDIA/cub && \
#     make EXTRA_NVCCFLAGS="-I./cub/cub" EXTRA_CCFLAGS="-fPIC" && \
#     cp yukarin_autoreg_cpp.so /app/src_cython/libyukarin_autoreg_cpp.so && \
#     cd - && rm -rf yukarin_autoreg_cpp

# RUN cd /app/src_cython && \
#     CFLAGS="-I." LDFLAGS="-L." python setup.py install
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/src_cython"

# install requirements
COPY requirements.txt /app/
COPY requirements_dev.txt /app/
RUN pip install -r <(cat requirements.txt | grep -v -x 'torch<2.0.0') -r requirements_dev.txt
