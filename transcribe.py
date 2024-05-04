import argparse
import os
import sys
import time
from typing import List

import numpy as np
import librosa
from mir_eval.util import midi_to_hz

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from onsets_and_frames import *


def float_samples_to_int16(y):
    """Convert floating-point numpy array of audio samples to int16."""
    # From https://github.com/tensorflow/magenta/blob/671501934ff6783a7912cc3e0e628fd0ea2dc609/magenta/music/audio_io.py#L48
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError('input samples not floating-point')
    return (y * np.iinfo(np.int16).max).astype(np.int16)


def load_and_process_audio(flac_path, sequence_length, device):
    random = np.random.RandomState(seed=42)

    audio, sr = librosa.load(flac_path, sr=SAMPLE_RATE)
    audio = float_samples_to_int16(audio)

    assert sr == SAMPLE_RATE
    assert audio.dtype == 'int16'

    audio = torch.ShortTensor(audio)

    if sequence_length is not None:
        audio_length = len(audio)
        step_begin = random.randint(audio_length - sequence_length) // HOP_LENGTH
        n_steps = sequence_length // HOP_LENGTH

        begin = step_begin * HOP_LENGTH
        end = begin + sequence_length

        audio = audio[begin:end].to(device)
    else:
        audio = audio.to(device)

    audio = audio.float().div_(32768.0)

    return audio


def transcribe(model, audio):
    mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

    predictions = {
        'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
        'offset': offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
        'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
        'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
    }

    return predictions


def transcribe_file(model_file: str, audio_paths: List[str], save_path: str, sequence_length: int,
                    onset_threshold: float, frame_threshold: float, device: str):
    model: OnsetsAndFrames = torch.load(model_file, map_location=device).eval()
    summary(model)

    for i, audio_path in enumerate(audio_paths):
        print(f'{i + 1}/{len(audio_paths)}: Processing {audio_path}...', file=sys.stderr)
        audio = load_and_process_audio(audio_path, sequence_length, device)
        predictions: dict = transcribe(model, audio)

        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'],
                                            onset_threshold, frame_threshold)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        os.makedirs(save_path, exist_ok=True)
        pred_path = os.path.join(save_path, os.path.basename(audio_path) + '.pred.png')
        save_pianoroll(pred_path, predictions['onset'], predictions['frame'])
        midi_path = os.path.join(save_path, os.path.basename(audio_path) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)


class Watcher:
    def __init__(self, monitor_path: str, args: argparse.Namespace):
        self.observer = Observer()
        self.monitor_path = monitor_path
        self.args = args

    def run(self):
        event_handler = NewRecordingHandler(self.args.model_file, self.args.save_path, self.args.sequence_length,
                                            self.args.onset_threshold, self.args.frame_threshold, self.args.device)
        self.observer.schedule(event_handler, self.monitor_path, recursive=True)
        self.observer.start()
        print(f'Watching {self.monitor_path}...')
        try:
            while True:
                time.sleep(1)
        finally:
            self.observer.stop()
            self.observer.join()


class NewRecordingHandler(FileSystemEventHandler):

    def __init__(self, model_file, save_path: str, sequence_length: int, onset_threshold: float, frame_threshold: float,
                 device: str):
        self.model_file = model_file
        self.save_path = save_path
        self.sequence_length = sequence_length
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.device = device

    def on_created(self, event):
        transcribe_file(self.model_file, [event.src_path], self.save_path, self.sequence_length, self.onset_threshold,
                        self.frame_threshold, self.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('--audio_paths', type=str, nargs='+')
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # This argument cannot be used in conjunction with audio_paths
    parser.add_argument('--monitor-directory', default=None, type=str)

    # todo add option to remove or retain files in output directory

    # todo add option to remove the files from input in watcher mode

    # todo add option to add folders to input in watcher mode -> folder naming + structure etc. is retained in output

    with torch.no_grad():
        """
        torch.no_grad() is useful for inference (not calling backward propagation)
        """
        if parser.parse_args().monitor_directory is not None:
            # process all files which are currently in the directory
            monitor_directory = parser.parse_args().monitor_directory
            for f in os.listdir(monitor_directory):
                if os.path.isfile(os.path.join(monitor_directory, f)):
                    transcribe_file(parser.parse_args().model_file, [os.path.join(monitor_directory, f)],
                                    parser.parse_args().save_path,
                                    parser.parse_args().sequence_length, parser.parse_args().onset_threshold,
                                    parser.parse_args().frame_threshold, parser.parse_args().device)
            # watch the directory for new future files (which are copied/moved into this dir)
            Watcher(parser.parse_args().monitor_directory, parser.parse_args()).run()
        else:
            transcribe_file(**vars(parser.parse_args()))
