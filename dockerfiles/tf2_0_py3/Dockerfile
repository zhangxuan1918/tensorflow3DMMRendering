ARG CUDA_BASE_VERSION
ARG UBUNTU_VERSION
ARG CUDNN_VERSION
ARG PYTHON_VERSION

# use CUDA + OpenGL
FROM nvidia/cudagl:${CUDA_BASE_VERSION}-devel-ubuntu${UBUNTU_VERSION}
MAINTAINER Domhnall Boyle (domhnallboyle@gmail.com)

# python3.7

ARG CUDA_BASE_VERSION
ARG UBUNTU_VERSION
ARG CUDNN_VERSION
ARG TENSORFLOW_VERSION
ARG GITHUB_PAT

# set environment variables
ENV CUDA_BASE_VERSION=${CUDA_BASE_VERSION}
ENV CUDNN_VERSION=${CUDNN_VERSION}
ENV TENSORFLOW_VERSION=${TENSORFLOW_VERSION}

#RUN echo $CUDA_BASE_VERSION
#RUN echo $CUDNN_VERSION
#RUN echo $TENSORFLOW_VERSION
#RUN echo $(echo $CUDNN_VERSION)-1+cuda$(echo $CUDA_BASE_VERSION)
#RUN echo $CUDNN_VERSION-1+cuda$CUDA_BASE_VERSION
#RUN echo $(echo $CUDNN_VERSION)-1+cuda$(echo $CUDA_BASE_VERSION)
#RUN echo $CUDNN_VERSION-1+cuda$CUDA_BASE_VERSION

# install apt dependencies
RUN apt-get update && apt-get install -y \
	git \
	vim \
	wget \
	software-properties-common \
	curl

# install python3.7 and pip
RUN apt-add-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 && \
    ln -s /usr/bin/python3.7 /usr/bin/python && \
    curl https://bootstrap.pypa.io/get-pip.py | python

# install newest cmake version
RUN apt-get purge cmake && cd ~ && wget https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5.tar.gz && tar -xvf cmake-3.14.5.tar.gz
RUN cd ~/cmake-3.14.5 && ./bootstrap && make && make install

# setting up cudnn
RUN apt-get install -y --no-install-recommends \
	libcudnn7=$CUDNN_VERSION-1+cuda$CUDA_BASE_VERSION \
	libcudnn7-dev=$CUDNN_VERSION-1+cuda$CUDA_BASE_VERSION
RUN apt-mark hold libcudnn7 && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN python -m pip install tensorflow-gpu==$TENSORFLOW_VERSION

# install dirt
ENV CUDAFLAGS='-DNDEBUG=1'
RUN cd ~ && git clone https://github.com/zhangxuan1918/dirt.git && \
 	python -m pip install dirt/

# run dirt test command
RUN python ~/dirt/tests/square_test.py

# install tensorflow-graphics
RUN RUN pip install git+https://zhangxuan1918:$GITHUB_PAT@git@github.com:zhangxuan1918/graphics.git

# install requirement
ADD requirements.txt .
RUN python -m pip install -r requirements.txt
RUN rm requirements.txt