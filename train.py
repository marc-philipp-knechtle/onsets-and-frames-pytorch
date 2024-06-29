import os
import sys
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *
from onsets_and_frames.dataset import PianoRollAudioDataset, SchubertWinterreiseDataset

ex = Experiment('train_transcriber')


@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO'

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

    ex.observers.append(FileStorageObserver.create(logdir))


def training_process(batch_size: int, checkpoint_interval: int, clip_gradient_norm: int,
                     device: str, iterations: int, learning_rate: float, learning_rate_decay_rate: float,
                     learning_rate_decay_steps: int, leave_one_out: bool, logdir: str, model_complexity: int,
                     resume_iteration: bool, sequence_length: int, train_on: str, validation_interval: int,
                     validation_length: int, writer: SummaryWriter, ):
    train_groups, validation_groups = ['train'], ['validation']
    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]
    dataset_training: Dataset
    validation_dataset: Dataset
    if train_on == 'MAESTRO':
        dataset_training = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    elif train_on == 'Winterreise':
        # HU33 and SC06 are intended for testing because they are the public ones
        dataset_training = SchubertWinterreiseDataset(groups=['FI55', 'FI66', 'FI80', 'OL06', 'QUI98', 'TR99'],
                                                      sequence_length=sequence_length)
        validation_dataset = SchubertWinterreiseDataset(groups=['AL98'], sequence_length=sequence_length)
    elif train_on == 'MAESTRO+Winterreise':
        maestro_training = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        maestro_validation = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
        winterreise_training = SchubertWinterreiseDataset(groups=['FI55', 'FI66', 'FI80', 'OL06', 'QUI98', 'TR99'],
                                                          sequence_length=sequence_length)
        winterreise_validation = SchubertWinterreiseDataset(groups=['AL98'], sequence_length=sequence_length)
        dataset_training = ConcatDataset([maestro_training, winterreise_training])
        validation_dataset = ConcatDataset([maestro_validation, winterreise_validation])
    elif train_on == 'all':
        # todo
        dataset_training = None
        validation_dataset = None
    else:
        dataset_training = MAPS(
            groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
            sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)
    loader = DataLoader(dataset_training, batch_size, shuffle=True, drop_last=True)
    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))
    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
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
                for key, value in evaluate(validation_dataset, model).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    try:
        training_process(batch_size, checkpoint_interval, clip_gradient_norm, device, iterations, learning_rate,
                         learning_rate_decay_rate, learning_rate_decay_steps, leave_one_out, logdir, model_complexity,
                         resume_iteration, sequence_length, train_on, validation_interval, validation_length, writer)
    except Exception as e:
        writer.add_text('train/error', str(e))
        print(e)
        print(str(e), file=sys.stderr)
