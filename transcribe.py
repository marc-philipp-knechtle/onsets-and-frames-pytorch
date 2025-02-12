import argparse
import os
import shutil
import sys
import time
from typing import List

import numpy as np
import librosa
from mir_eval.util import midi_to_hz
from torch import Tensor, ShortTensor

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from onsets_and_frames import *


def float_samples_to_int16(y):
    """Convert floating-point numpy array of audio samples to int16."""
    # From https://github.com/tensorflow/magenta/blob/671501934ff6783a7912cc3e0e628fd0ea2dc609/magenta/music/audio_io.py#L48
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError('input samples not floating-point')
    return (y * np.iinfo(np.int16).max).astype(np.int16)


def load_and_process_audio(flac_path, sequence_length, device, duration=None) -> ShortTensor:
    """
    Args:
        flac_path:
        sequence_length:
        device:
        duration: Duration in seconds where the file is loaded -> This is passed to librosa such that it works

    Returns: ShortTensor of the audio
    """
    random = np.random.RandomState(seed=42)

    audio, sr = librosa.load(flac_path, sr=SAMPLE_RATE, duration=duration)
    audio = float_samples_to_int16(audio)

    assert sr == SAMPLE_RATE
    assert audio.dtype == 'int16'

    audio: ShortTensor = torch.ShortTensor(audio)

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


def transcribe(model: OnsetsAndFrames, audio: Tensor):
    melspect = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)

    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(melspect)

    predictions = {
        'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
        'offset': offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
        'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
        'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
    }

    return predictions


def transcribe_file(model_file: str, audio_paths: List[str], save_path: str, sequence_length: int,
                    onset_threshold: float, frame_threshold: float, device: str, show_summary: bool = True):
    torch.cuda.empty_cache()

    model: OnsetsAndFrames = torch.load(model_file, map_location=device).eval()
    if show_summary:
        summary(model)

    for i, audio_path in enumerate(audio_paths):
        print(f'{i + 1}/{len(audio_paths)}: Processing {audio_path}...', file=sys.stderr)
        audio = load_and_process_audio(audio_path, sequence_length, device, duration=None)
        try:
            predictions: dict = transcribe(model, audio)
        except RuntimeError:
            print("Loading a short duration of the clip to provide context")
            print("Emptying Cache:")
            torch.cuda.empty_cache()
            print("Called torch.cuda.empty_cache()")
            audio = load_and_process_audio(audio_path, sequence_length, device, duration=5)
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


def duplicate_directory_structure(src: str, dst: str):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Walk through the source directory
    for dirpath, dirnames, filenames in os.walk(src):
        # Construct the destination path
        structure = os.path.join(dst, os.path.relpath(dirpath, src))

        # Create the directory structure in the destination path
        if not os.path.exists(structure):
            os.makedirs(structure)


def remove_prefix(text: str, prefix: str):
    """
    Args:
        text: some string text
        prefix: the prefix to remove from this text. The prefix has to match exactly the text which will be removed.
    Returns:
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def count_files_recursively(dir_path: str) -> int:
    """
    Args:
        dir_path: directory path which is traversed recursively
    Returns: the number of all files in this directory
    """
    total_files: int = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        total_files += len(filenames)
    return total_files


def transcribe_dir(model_file: str, directory_to_transcribe: str, save_path: str, sequence_length: int,
                   onset_threshold: float,
                   frame_threshold: float, device: str, remove_input: bool = False):
    """
    This is an adapted version of transcribe_file. The goal of this method is to retain the directory structure.
    Sometimes, there are naming conventions included in the naming of the directories. This method is intended to retain
    them.
    IMPORTANT DIFFERENCE!
    The transcribe_dir method removes the processed files directly after they are finished processing!
    Args:
        remove_input:
        model_file:
        directory_to_transcribe:
        save_path:
        sequence_length:
        onset_threshold:
        frame_threshold:
        device:
    """
    duplicate_directory_structure(directory_to_transcribe, save_path)
    # This is the directory name of the input. We save this to prune the name from the path
    # This is required to duplicate the directory for the output
    input_dirname: str = directory_to_transcribe

    for root, dirs, files in os.walk(directory_to_transcribe):
        # This method walks through each directory in the passed directory
        # For each directory, recursively, this lists the dirs and files in this directory
        root_without_input = remove_prefix(root, input_dirname)
        root_without_input = remove_prefix(root_without_input, os.sep)
        for filename in files:
            try:
                transcribe_file(model_file, [os.path.join(root, filename)],
                                os.path.join(save_path, root_without_input),
                                sequence_length,
                                onset_threshold, frame_threshold, device, show_summary=False)
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting... \n")
                print(f"The input file will be retained, NOT REMOVED! ({os.path.join(root, filename)})")
                print("Please check the output manually if there are any files to clean up!")
                sys.exit(0)
            if remove_input:
                print(f"Finished processing: {os.path.join(root, filename)}, removing file from input directory!")
                os.remove(os.path.join(root, filename))


class Watcher:
    def __init__(self, monitor_path: str, args: argparse.Namespace):
        self.observer = Observer()
        self.monitor_path = monitor_path
        self.args = args

    def run(self):
        event_handler = NewRecordingHandler(self.args.model_file, self.args.save_path, self.args.sequence_length,
                                            self.args.onset_threshold, self.args.frame_threshold, self.args.device,
                                            self.args.monitor_directory)
        self.observer.schedule(event_handler, self.monitor_path, recursive=False)
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
                 device: str, monitor_directory: str):
        self.model_file = model_file
        self.save_path = save_path
        self.sequence_length = sequence_length
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.device = device
        self.monitor_directory = monitor_directory

    def on_created(self, event):
        if os.path.isfile(event.src_path):
            transcribe_file(self.model_file, [event.src_path], self.save_path, self.sequence_length,
                            self.onset_threshold,
                            self.frame_threshold, self.device)
            os.remove(event.src_path)
        elif os.path.isdir(event.src_path):
            # Calling transcribe_dir on monitor_directory and not on event.src_path!
            # This is because running on event.src_path could lead to the method beeing called from subdirectories
            # This causes the directory creation mechanism to fail
            # Maybe it would be an option to discard the directory creation mechanism
            transcribe_dir(self.model_file, self.monitor_directory, self.save_path, self.sequence_length,
                           self.onset_threshold,
                           self.frame_threshold, self.device)
            # not using os.rmdir() because this does not work for non-empty directories!
            shutil.rmtree(event.src_path)
        else:
            print("Reaching other type of handling compared to isfile and isdir")
            print(event)
            raise RuntimeError("Handling of events other than file or directory is not supported!")


def check_output_directory(output_directory: str, clear_output: bool):
    if len(output_directory) and clear_output:
        # Ensure that the output directory by itself is not removed, only the directories within that output directory
        user_input: str = ""
        if count_files_recursively(output_directory) >= 10:
            user_input = input("Detected more than 10 files to be deleted from the output (--clear-output):\n"
                               "Do you really want to continue? [y/n]")
        if user_input == "y":
            for d in os.listdir(output_directory):
                shutil.rmtree(os.path.join(output_directory, d))
            print(f"Cleared output directory: {output_directory}")
        else:
            raise RuntimeError(
                "Stopping the execution of the application. --clear-output specified without reassurement.")


def handle_file_or_directory(path: str, args: argparse.Namespace):
    """
    Args:
        path: path is not extracted from args variable to allow for differentiation audio_paths and monitor_directory
        args: argparse args
    """
    # This is required, because it might happen that a filepath is passed to this function
    dirpath = os.path.dirname(path) if os.path.isfile(path) else path
    for f in os.listdir(dirpath):
        if os.path.isfile(os.path.join(dirpath, f)):
            transcribe_file(args.model_file, [os.path.join(dirpath, f)],
                            args.save_path,
                            args.sequence_length, args.onset_threshold,
                            args.frame_threshold, args.device)
            if args.remove_input:
                os.remove(os.path.join(dirpath, f))
        if os.path.isdir(os.path.join(dirpath, f)):
            transcribe_dir(args.model_file, path, args.save_path, args.sequence_length,
                           args.onset_threshold, args.frame_threshold, args.device, args.remove_input)
            if args.remove_input:
                for directory in os.listdir(path):
                    shutil.rmtree(os.path.join(path, directory))


def main(args: argparse.Namespace):
    check_output_directory(args.save_path, args.clear_output)
    # todo add option to remove the files from input in watcher mode
    # todo adding progress bar in non-monitoring mode
    # todo adding removal of the files once ready when processing when in directory mode
    # -> easier debugging when sth goes wrong
    # -> it's possible to rerun some predictions without
    # todo adding loguru
    with torch.no_grad():
        """
        torch.no_grad() is useful for inference (not calling backward propagation)
        """
        if args.audio_paths is not None and args.monitor_directory is not None:
            raise RuntimeError("Specified arguments audio_paths and monitor_directory are mutually exclusive.")
        elif args.monitor_directory is not None:  # = watcher mode is enabled
            # process all files which are currently in the directory
            handle_file_or_directory(args.monitor_directory, args)
            # watch the directory for new future files (which are copied/moved into this dir)
            Watcher(parser.parse_args().monitor_directory, parser.parse_args()).run()
        elif args.audio_paths is not None:
            for path in args.audio_paths:
                handle_file_or_directory(path, args)
        else:
            raise RuntimeError("You have to either specify audio_paths or monitor_directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('--audio_paths', type=str, nargs='+', default=None)
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear-output', type=bool, default=False)
    parser.add_argument('--remove-input', type=bool, default=False)

    # This argument cannot be used in conjunction with audio_paths
    parser.add_argument('--monitor-directory', default=None, type=str)

    arguments: argparse.Namespace = parser.parse_args()

    main(arguments)
