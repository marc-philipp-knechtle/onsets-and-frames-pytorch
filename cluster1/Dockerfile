# FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
# RuntimeError: cannot cache function '__shear_dense': no locator available for file '/opt/conda/envs/onsets-and-frames-pytorch/lib/python3.7/site-packages/librosa/util/utils.py'
# https://github.com/numba/numba/issues/5566
# Running this led to this error described above. This has sth to do with numba and seems not resolvable
# I tried using different Cache Environment Variables
# Also trying out different types of numba and librosa versions
# All off these attempts were unsuccessful :(
# Therefore trying to switch to a different pytorch cuda version combination
# FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
# -> same error as above
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

COPY ../environment.yml .

# Required because of issue with prompt asking for geographic location
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1-dev # this is required for the python package soundfile

# duplicate to the pytorch dockerfile definition (just as a reminder)
WORKDIR /workspace

RUN conda env create -f environment.yml
# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "onsets-and-frames-pytorch", "/bin/bash", "-c"]

COPY onsets_and_frames onsets_and_frames
COPY train.py .
COPY evaluate.py .


# RUN mkdir /workspace/data/cache
# RUN export NUMBA_CACHE_DIR=/workspace/cache
# RUN chown root $NUMBA_CACHE_DIR
# RUN mkdir /workspace/runs
RUN mkdir /tmp/numba_cache & chmod 777 /tmp/numba_cache & NUMBA_CACHE_DIR=/tmp/numba_cache

# Testing if the container has all dependencies
RUN conda run -n onsets-and-frames-pytorch python3 -c "import torch"

# ENTRYPOINT ["conda", "run", "-n", "onsets-and-frames-pytorch", "python3", "train.py"]

# Set a dummy entrypoint for debugging
# ENTRYPOINT ["/bin/bash"]
