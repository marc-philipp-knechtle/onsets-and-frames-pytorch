# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# .venv: Python 3.8.19
# conda: Python 3.7.16

# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# --> python 3.10 -> to recent python version
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
# --> Led to some form of timeout
# pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# --> python 3.10.13
# pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# --> python 3.10.11
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# --> python 3.10.8
# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# --> python 3.8
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
# --> python 3.7.11

# FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
# -> This led to error: (I Forgot Error)

# cluster2 is an attempt to create a k8s job without dealing with conda

# Required because of issue with prompt asking for geographic location
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1-dev # this is required for the python package soundfile
RUN apt-get install -y git

# duplicate to the pytorch dockerfile definition (just as a reminder)
WORKDIR /workspace

COPY cluster2/requirements.txt .
RUN conda install pip
RUN conda install GitPython
RUN pip install -r requirements.txt
# todo why do I have this here? I think this is some weird caching issue
RUN pip install pandas

COPY onsets_and_frames onsets_and_frames
COPY train.py .
COPY evaluate.py .

RUN mkdir /tmp/numba_cache & chmod 777 /tmp/numba_cache & NUMBA_CACHE_DIR=/tmp/numba_cache

# Set a dummy entrypoint for debugging
# ENTRYPOINT ["/bin/bash"]