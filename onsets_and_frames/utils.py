import sys
from functools import reduce
from typing import List

import matplotlib.pyplot as plt
from matplotlib import patches
import pretty_midi
import torch
from PIL import Image
from torch.nn.modules.module import _addindent

from onsets_and_frames import HOP_LENGTH, SAMPLE_RATE, MIN_MIDI


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    """
    Print a summary of the model
    Args:
        model: model file
        file: output file
    Returns: Nothing
    """

    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count


def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    onsets = (1 - (onsets.t() > onset_threshold).to(torch.uint8)).cpu()
    frames = (1 - (frames.t() > frame_threshold).to(torch.uint8)).cpu()
    both = (1 - (1 - onsets) * (1 - frames))
    image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)


def save_pianoroll_matplotlib(midifile: pretty_midi.PrettyMIDI, save_path: str, start_param: int = None,
                              end_param: int = None,
                              onset_prediction: torch.Tensor = None):
    # idea: create pianoroll representation for standard midi data
    # -> create separate pianoroll representation considering the results of the onset detector.
    instruments: List[pretty_midi.Instrument] = midifile.instruments
    if len(instruments) > 1:
        raise RuntimeError('Encountered unexpected number of instruments. This method is only intended for one'
                           'instrument only midi files.')
    if start_param is None and end_param is None or start_param is not None and end_param is None:
        raise RuntimeError('You cannot set start param without setting end param and vice versa.'
                           'This feature is currently not supported.')
    instrument: pretty_midi.Instrument = instruments[0]

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 1, 1)

    pitch_min: int = 1000
    pitch_max: int = 0
    starts = []
    ends = []

    pitches_during_considered_period = []

    note: pretty_midi.Note
    for note in instrument.notes:
        pitch = note.pitch
        start = note.start
        end = note.end
        duration = note.duration

        if start_param is not None and end_param is not None:
            if start > start_param and end < end_param:
                pitches_during_considered_period.append(pitch)
        else:
            if pitch > pitch_max: pitch_max = pitch
            if pitch < pitch_min: pitch_min = pitch
            starts.append(start)
            ends.append(end)

        rect = patches.Rectangle(
            (start, pitch - 0.5), duration, 1, linewidth=0.35, edgecolor='k', alpha=1)
        ax.add_patch(rect)

    if onset_prediction is not None:
        # see decoding.py for comments on this approach
        onsets = (onset_prediction > 0.5).cpu().to(torch.uint8)
        onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1
        for nonzero in onset_diff.nonzero():
            frame = nonzero[0].item()
            pitch = nonzero[1].item() + MIN_MIDI
            scaling = HOP_LENGTH / SAMPLE_RATE
            time = frame * scaling
            rect = patches.Rectangle(
                (time, pitch - 0.5), 0.03, 1, linewidth=0.35, edgecolor='k', alpha=1, facecolor='red')
            ax.add_patch(rect)

    if start_param is None and end_param is None:
        ax.set_xlim([min(starts), max(ends) + 0.5])
        ax.set_ylim([pitch_min - 1.5, pitch_max + 1.5])
    else:
        ax.set_xlim([start_param, end_param + 0.5])
        ax.set_ylim([min(pitches_during_considered_period) - 1, max(pitches_during_considered_period) + 1])
    ax.grid()
    # ticks and gridlines are below all artists
    ax.set_axisbelow(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch')

    fig.savefig(save_path, dpi=900)
