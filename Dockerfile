FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

COPY ./environment.yml .

RUN apt-get update && \
    apt-get install -y \
    libsndfile1-dev # this is required for the python package soundfile

RUN conda env create -f environment.yml
# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "onsets-and-frames-pytorch", "/bin/bash", "-c"]

COPY onsets_and_frames onsets_and_frames/
COPY train.py .
COPY evaluate.py .

# Testing if the container has all dependencies
RUN conda run -n onsets-and-frames-pytorch python3 -c "import torch"

ENTRYPOINT ["conda", "run", "-n", "onsets-and-frames-pytorch", "python3", "train.py"]

# Set a dummy entrypoint for debugging
# ENTRYPOINT ["/bin/bash"]
