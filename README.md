# PyTorch Implementation of Onsets and Frames

This is a [PyTorch](https://pytorch.org/) implementation of
Google's [Onsets and Frames](https://magenta.tensorflow.org/onsets-frames) model, using
the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro) for training and the Disklavier portion of
the [MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/)
for testing.

This fork from the original repo includes further information on running and testing these systems.
This fork is a combination of
the [Original Pytorch Onset and Frames Implementation](https://github.com/jongwook/onsets-and-frames)
as well as a [Pull request including a pretrained model](https://github.com/jongwook/onsets-and-frames/pull/18).

Furthermore, there are some additions (ordered by importance):

* Adding directory handling and monitoring to [transcribe.py](transcribe.py). This is especially important for the
  handling of nested directories -> The structure is retained in the output
* switching to conda environment (environment.yaml)

## Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory is recommended.

### Creating the Environment

To simplify recreation, we provide an `environment.yml` for conda.
It might need some editing for the `name` and `prefix` values.

There is also a python virtual environment with the corresponding `requirements.txt`. This was mainly used for docker
testing. I assume that some of the cuda memory issues originate from the use of conda, but I have to further verify
this.

```bash
conda env create -f environment.yaml
```

### Downloading Dataset

The `data` subdirectory already contains the MAPS database. To download the Maestro dataset, first make sure that you
have `ffmpeg` executable and run `prepare_maestro.sh` script:

```bash
ffmpeg -version
cd data
./prepare_maestro.sh
```

This will download the full Maestro dataset from Google's server and automatically unzip and encode them as FLAC files
in order to save storage. However, you'll still need about 200 GB of space for intermediate storage.

### Training

```bash
python train.py
```

`train.py` is written using [sacred](https://sacred.readthedocs.io/), and accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

Trained models will be saved in the specified `logdir`, otherwise at a timestamped directory under `runs/`.

### Testing

To evaluate the trained model using the MAPS database, run the following command to calculate the note and frame
metrics:

```bash
python evaluate.py runs/model/model-100000.pt
```

Specifying `--save-path` will output the transcribed MIDI file along with the piano roll images:

```bash
python evaluate.py runs/model/model-100000.pt --save-path output/
```

In order to test on the Maestro dataset's test split instead of the MAPS database, run:

```bash
python evaluate.py runs/model/model-100000.pt MAESTRO test
```

You can download a pretrained
model [here](https://drive.google.com/file/d/1Mj2Em07Lvl3mvDQCCxOYHjPiB-S0WGT1/view?usp=sharing) and run `transcribe.py`
to transcribe piano audio files:

```bash
python transcribe.py model-500000.pt <path to audio files> --save-path output/
```

## Docker Container

### Local Testing

Building:

```bash
docker build -t onsets-and-frames-pytorch:1.0 cluster2/Dockerfile
```

Running:

```bash
docker run --name onsets-and-frames-pytorch --gpus all  -v /media/mpk/external-nvme/onsets-and-frames-pytorch/data/MAESTRO:/workspace/data/MAESTRO -v /media/mpk/external-nvme/onsets-and-frames-pytorch/runs:/workspace/runs -t onsets-and-frames-pytorch:1.0
```

The Docker container is mainly used to run the training and evaluation on the university infrastructure. However
it can also be used for local testing.

## Implementation Details

This implementation contains a few of the additional improvements on the model that were reported in the Maestro paper,
including:

* Offset head
* Increased model capacity, making it 26M parameters by default
* Gradient stopping of inter-stack connections
* L2 Gradient clipping of each parameter at 3
* Using the HTK mel frequencies

Meanwhile, this implementation does not include the following features:

* Variable-length input sequences that slices at silence or zero crossings
* Harmonically decaying weights on the frame loss

Despite these, this implementation is able to achieve a comparable performance to what is reported on the Maestro paper
as the performance without data augmentation.
