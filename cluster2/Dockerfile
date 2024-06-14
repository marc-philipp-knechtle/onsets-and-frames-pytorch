FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
# .venv: Python 3.8.19
# conda: Python 3.7.16

# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# python 3.10 -> to recent python version
# FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
# -> This led to error: (I Forgot Error)

# cluster2 is an attempt to create a k8s job without dealing with conda

RUN apt-get update && \
    apt-get install -y \
    libsndfile1-dev # this is required for the python package soundfile


# Set a dummy entrypoint for debugging
ENTRYPOINT ["/bin/bash"]