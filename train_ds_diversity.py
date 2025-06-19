import glob
import logging
import re
import os
import sys
from datetime import datetime
import random
from typing import List, Tuple, Dict

import numpy as np
from caffe2.python.net_printer import print_task_group
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate, evaluate_ap
from onsets_and_frames import *
from onsets_and_frames.dataset import SchubertWinterreiseDataset, SchubertWinterreiseVoice, SchubertWinterreisePiano, \
    PianoRollAudioDataset

from onsets_and_frames.dataset import dataset_definitions as ddef
from onsets_and_frames.earlystopping import EarlyStopping
from train import create_model, run_iteration

import train

ex = Experiment('train_transcriber')

training_configs = {
    'MuN': lambda: [ddef['MuN_train']()],
    'MuN+B10': lambda: [ddef['MuN_train'](), ddef['b10_train']()],
    'MuN+B10+PhA': lambda: [ddef['MuN_train'](), ddef['b10_train'](), ddef['PhA_train']()],
    'MuN+B10+PhA+CSD': lambda: [ddef['MuN_train'](), ddef['b10_train'](), ddef['PhA_train'](), ddef['CSD_train']()],
    'all': lambda: [ddef['MuN_train'](), ddef['b10_train'](), ddef['PhA_train'](), ddef['CSD_train'](),
                    ddef['winterreise_training']()],
}

training_run_capacities = {
    'MuN': {'MusicNet': 288},
    'MuN+B10': {'MusicNet': 284, 'Bach10': 8},
    'MuN+B10+PhA': {'MusicNet': 282, 'Bach10': 4, 'PhenicxAnechoic': 8},
    'MuN+B10+PhA+CSD': {'MusicNet': 272, 'Bach10': 4, 'PhenicxAnechoic': 2, 'ChoralSinginDataset': 10},
    'all': {'MusicNet': 207, 'Bach10': 4, 'PhenicxAnechoic': 2, 'ChoralSingingDataset': 5, 'SchubertWinterreiseDataset': 65},
}


@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000  # -> epochs=iterations/batch_size = 500000/8 = 62500
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO'
    data_path = 'data/MAESTRO'

    batch_size = 8
    sequence_length = 327680
    """
    This is the length of the input sequence during training. In seconds: ..../SAMPLE_RATE = 20.48s
    """
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        print(
            f'total memory available from cuda: '
            f'{torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3):.2f}GB\n'
            f'This is smaller than required: {10e9 / (1024 ** 3):.2f}GB')
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    clear_computed: bool = False

    ex.observers.append(FileStorageObserver.create(logdir))


def training_process(batch_size: int, checkpoint_interval: int, clip_gradient_norm: int,
                     device: str, iterations: int, learning_rate: float, learning_rate_decay_rate: float,
                     learning_rate_decay_steps: int, leave_one_out: bool, logdir: str, model_complexity: int,
                     resume_iteration: bool, sequence_length: int, train_on: str, data_path: str,
                     validation_interval: int, validation_length: int, writer: SummaryWriter, clear_computed: bool):
    dataset_training: List[Dataset] = training_configs[train_on]()
    dataset_validation = ConcatDataset([
        ddef['MuN_validation'](),
        ddef['winterreise_validation'](),
        ddef['b10_validation'](),
        ddef['CSD_validation'](),
    ])

    train_dsets = []

    ds: PianoRollAudioDataset
    for ds in dataset_training:
        allowed_length = training_run_capacities[train_on][str(ds)]
        if len(ds) <= allowed_length:
            ds_indices = []
            while len(ds_indices) + len(ds) < allowed_length:
                ds_indices.extend(range(0, len(ds)))
            ds_indices.extend(random.sample(range(0, len(ds)), allowed_length - len(ds_indices)))
        else:
            ds_indices = random.sample(range(0, len(ds)), allowed_length)
        subset = torch.utils.data.Subset(ds, ds_indices)
        train_dsets.append(subset)

    loader = DataLoader(ConcatDataset(train_dsets), batch_size, drop_last=True, shuffle=True)

    model, optimizer, resume_iteration = create_model(device, learning_rate, logdir, model_complexity, resume_iteration)
    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    early_stopping = EarlyStopping()
    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    try:
        batch: Dict
        """
        Dict with:
        path
        audio
        label
        velocity
        onset
        offset
        frame
        """
        for i, batch in zip(loop, cycle(loader)):
            """
            zip(loop, cycle(loader)) -> pairs values from loop with the loader()
            cyle(loader) -> infinite loop over the loader, starts from beginning if the loader is exhausted
            
            We need this mechanism to create the batches on the fly, because otherwise this would not fit into gpu memory.
            Test this by: 
            asdf = list(zip(loop, cycle(loader)))
            -> memory error... 
            """
            run_iteration(batch, checkpoint_interval, clip_gradient_norm, i, logdir, model, optimizer, scheduler,
                          dataset_validation, validation_interval, writer, early_stopping)
            if early_stopping.early_stop:
                logging.info(f'EARLY STOPPING! saving mode early-stopping-model-{i}.pt')
                torch.save(early_stopping.best_model_state, os.path.join(logdir, f'early-stopping-model-{i}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
                break
    except Exception as e:
        raise e
    finally:
        if clear_computed:
            train.clear_train_val_ds(dataset_training, dataset_validation)

@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, data_path, batch_size,
          sequence_length, model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate,
          leave_one_out, clip_gradient_norm, validation_length, validation_interval, clear_computed):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    try:
        training_process(batch_size, checkpoint_interval, clip_gradient_norm, device, iterations, learning_rate,
                         learning_rate_decay_rate, learning_rate_decay_steps, leave_one_out, logdir, model_complexity,
                         resume_iteration, sequence_length, train_on, data_path, validation_interval, validation_length,
                         writer, clear_computed)
    except Exception as e:
        writer.add_text('train/error', str(e))
        logging.error(str(e))
        print(e)
        print(str(e), file=sys.stderr)
        raise e
