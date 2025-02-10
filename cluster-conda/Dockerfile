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
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# Error: RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
# # Error: RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
## >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<
#
#    Traceback (most recent call last):
#      File "/opt/conda/lib/python3.7/site-packages/conda/exceptions.py", line 1079, in __call__
#        return func(*args, **kwargs)
#      File "/opt/conda/lib/python3.7/site-packages/conda/cli/main.py", line 84, in _main
#        exit_code = do_call(args, p)
#      File "/opt/conda/lib/python3.7/site-packages/conda/cli/conda_argparse.py", line 83, in do_call
#        return getattr(module, func_name)(args, parser)
#      File "/opt/conda/lib/python3.7/site-packages/conda/cli/main_run.py", line 26, in execute
#        args.dev, args.debug_wrapper_scripts, call)
#      File "/opt/conda/lib/python3.7/site-packages/conda/utils.py", line 403, in wrap_subprocess_call
#        with Utf8NamedTemporaryFile(mode='w', prefix=tmp_prefix, delete=False) as fh:
#      File "/opt/conda/lib/python3.7/site-packages/conda/_vendor/auxlib/compat.py", line 83, in Utf8NamedTemporaryFile
#        dir=dir, delete=delete)
#      File "/opt/conda/lib/python3.7/tempfile.py", line 547, in NamedTemporaryFile
#        (fd, name) = _mkstemp_inner(dir, prefix, suffix, flags, output_type)
#      File "/opt/conda/lib/python3.7/tempfile.py", line 258, in _mkstemp_inner
#        fd = _os.open(file, flags, 0o600)
#    PermissionError: [Errno 13] Permission denied: '/opt/conda/envs/onsets-and-frames-pytorch/.tmp6xy4jja_'
#
#`$ /opt/conda/bin/conda run -n onsets-and-frames-pytorch python3 train.py`


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
