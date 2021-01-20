FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
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

# install requirements
COPY requirements.txt /app/
RUN pip install -r <(cat requirements.txt | grep -v -x 'torch')

# cpp
COPY src_cython /app/src_cython
RUN cd /app/src_cython && \
    CFLAGS="-I." LDFLAGS="-L." python setup.py install
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/src_cython"
