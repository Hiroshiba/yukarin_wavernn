FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu16.04
SHELL ["/bin/bash", "-c"]

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git gcc g++ cmake sudo zip unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda install -y python=3.7.5 cython numpy

RUN apt-get update && \
    apt-get install -y swig libsndfile1-dev libasound2-dev && \
    apt-get clean

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r <(cat requirements.txt | sed -e 's/cupy/cupy-cuda102>=7.7.0,<8.0.0/g')
COPY requirements-dev.txt /app/
RUN pip install -r requirements-dev.txt

# cpp
COPY src_cython /app/src_cython
RUN cd /app/src_cython && \
    curl https://raw.githubusercontent.com/Hiroshiba/yukarin_wavernn_cpp/cuda9.0/CppWaveRNN/CppWaveRNN.h > /app/src_cython/CppWaveRNN.h && \
    CFLAGS="-I." \
    LDFLAGS="-L." \
    python setup.py install
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/src_cython"

# optuna
RUN apt-get update && \
    apt-get install -y python3-dev libmysqlclient-dev && \
    apt-get clean && \
    pip install optuna mysqlclient

CMD bash
