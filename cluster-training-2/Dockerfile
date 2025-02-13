FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# Required because of issue with prompt asking for geographic location
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1-dev # this is required for the python package soundfile
RUN apt-get install -y git

WORKDIR /workspace

RUN conda install pip
RUN conda install GitPython

COPY cluster-training/requirements.txt .
RUN pip install -r /workspace/requirements.txt

RUN mkdir /tmp/numba_cache & chmod 777 /tmp/numba_cache & NUMBA_CACHE_DIR=/tmp/numba_cache
