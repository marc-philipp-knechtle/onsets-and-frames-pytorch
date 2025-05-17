import glob
import logging
import re
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *
from onsets_and_frames.dataset import SchubertWinterreiseDataset, SchubertWinterreiseVoice, SchubertWinterreisePiano, \
    PianoRollAudioDataset

from onsets_and_frames.dataset import dataset_definitions as ddef
from onsets_and_frames.transcriber import Frames
from train import create_datasets, create_sampler, EarlyStopping

ex = Experiment('train_frame')

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


def create_model(device, learning_rate, logdir, model_complexity, resume_iteration):
    if resume_iteration is None and "transcriber" in logdir:
        model = Frames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    elif 'transcriber' not in logdir:
        # "We want to resume the last iteration, however the resume iteration has not been set.
        # we have to determine the last iteration automatically"
        models = glob.glob(os.path.join(logdir, '*'))
        matching_files = [file for file in models if 'model-' in os.path.basename(file)]
        matching_files.sort()
        if len(matching_files) > 0:
            model_path: str = matching_files[-1]
            resume_iteration: int = int(re.findall(r'\d+', os.path.basename(model_path))[0])
            model = torch.load(model_path)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
            optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))
            logging.info(f"Resuming training at previously automatically determined state: {model_path}")
        else:
            model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
            resume_iteration = 0
            logging.info("Creating logdir automatically and beginning training from the start.")
    else:
        raise RuntimeError(f'Not implemented!!!')

    return model, optimizer, resume_iteration


def run_iteration(batch: Dict[str, Tensor], checkpoint_interval, clip_gradient_norm, i, logdir, model, optimizer, scheduler,
                  validation_dataset, validation_interval, writer: SummaryWriter, early_stopping: EarlyStopping):
    """

    Args:
        batch: -> Dict with Tensors as content. Each Tensor is in the first dimension len(batch)
                contents: path, audio, label, velocity, onset, offset, frame
        checkpoint_interval:
        clip_gradient_norm:
        i:
        logdir:
        model:
        optimizer:
        scheduler:
        validation_dataset:
        validation_interval:
        writer:
        early_stopping:

    Returns:

    """
    predictions, losses = model.run_on_batch(batch)
    loss = sum(losses.values())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if clip_gradient_norm:
        clip_grad_norm_(model.parameters(), clip_gradient_norm)
    for key, value in {'loss': loss, **losses}.items():
        writer.add_scalar(key, value.item(), global_step=i)
    if i % validation_interval == 0:
        model.eval()
        with torch.no_grad():
            eval_dct = evaluate(validation_dataset, model)
            for key, value in eval_dct.items():
                writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            early_stopping(np.mean(eval_dct['metric/frame/f1']), model)
        model.train()
    if i % checkpoint_interval == 0:
        logging.info(f'saving mode model-{i}.pt')
        torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))


def training_process(batch_size: int, checkpoint_interval: int, clip_gradient_norm: int,
                     device: str, iterations: int, learning_rate: float, learning_rate_decay_rate: float,
                     learning_rate_decay_steps: int, leave_one_out: bool, logdir: str, model_complexity: int,
                     resume_iteration: bool, sequence_length: int, train_on: str, data_path: str,
                     validation_interval: int, validation_length: int, writer: SummaryWriter, clear_computed: bool):

    dataset_training: ConcatDataset
    dataset_training, dataset_validation = create_datasets(sequence_length, train_on, data_path)

    if type(dataset_training) == ConcatDataset:
        sampler = create_sampler(dataset_training)
        loader = DataLoader(dataset_training, batch_size, drop_last=True, sampler=sampler)
    elif isinstance(dataset_training, PianoRollAudioDataset):
        loader = DataLoader(dataset_training, batch_size, drop_last=True, shuffle=True)
    else:
        raise RuntimeError(f'Unknown type of dataset: {str(dataset_training)}')

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
            if isinstance(dataset_training, ConcatDataset):
                dataset_impl: PianoRollAudioDataset
                for dataset_impl in dataset_training.datasets:
                    dataset_impl.clear_computed()
            elif isinstance(dataset_training, PianoRollAudioDataset):
                dataset_training.clear_computed()
            else:
                raise RuntimeError(
                    f'Expected Concat Dataset or PianoRollAudioDataset but got something else: {type(dataset_training)}')

            if isinstance(dataset_validation, ConcatDataset):
                dataset_impl: PianoRollAudioDataset
                for dataset_impl in dataset_validation.datasets:
                    dataset_impl.clear_computed()
            elif isinstance(dataset_validation, PianoRollAudioDataset):
                dataset_validation.clear_computed()
            else:
                raise RuntimeError(
                    f'Expected Concat Dataset or PianoRollAudioDataset but got something else: {type(dataset_validation)}')


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

