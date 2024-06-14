FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
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
# --> python 3.8.12
# FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
# -> This led to error: (I Forgot Error)

# cluster2 is an attempt to create a k8s job without dealing with conda

# Required because of issue with prompt asking for geographic location
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1-dev # this is required for the python package soundfile

WORKDIR /workspace

COPY cluster2/requirements.txt .
RUN conda install pip
RUN pip install -r requirements.txt

COPY onsets_and_frames onsets_and_frames
COPY train.py .
COPY evaluate.py .

# Set a dummy entrypoint for debugging
# ENTRYPOINT ["/bin/bash"]