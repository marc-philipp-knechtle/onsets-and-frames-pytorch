import argparse
import os
import shutil
import sys
import logging
from collections import defaultdict
from datetime import datetime
from typing import List

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm
from torch.utils.data import IterableDataset
import onsets_and_frames.dataset as dataset_module
from onsets_and_frames import *

eps = sys.float_info.epsilon


def evaluate(pianoroll_dataset: IterableDataset, model: OnsetsAndFrames, onset_threshold=0.5,
             frame_threshold=0.5, save_path=None) -> dict:
    metrics = defaultdict(list)

    for label in tqdm(pianoroll_dataset):
        prediction, losses = model.run_on_batch(label)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in prediction.items():
            value.squeeze_(0).relu_()

        """
        reference = ground truth value
        pitches
        intervals
        velocities
        """
        p_ref: np.ndarray
        i_ref: np.ndarray
        v_ref: np.ndarray
        p_ref, i_ref, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])
        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)
        """
        estimate = prediction
        pitches 
        intervals
        velocities
        """
        p_est, i_est, v_est = extract_notes(prediction['onset'], prediction['frame'], prediction['velocity'],
                                            onset_threshold, frame_threshold)
        t_est, f_est = notes_to_frames(p_est, i_est, prediction['frame'].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                  offset_ratio=None, velocity_tolerance=0.1)
        metrics['metric/note-with-velocity/precision'].append(p)
        metrics['metric/note-with-velocity/recall'].append(r)
        metrics['metric/note-with-velocity/f1'].append(f)
        metrics['metric/note-with-velocity/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
        metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(
            hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            dirname: str = os.path.basename(os.path.dirname(label['path']))
            label_path: str = str(
                os.path.join(save_path, dirname + '_' + os.path.basename(label['path']) + '.label.png'))
            save_pianoroll(label_path, label['onset'], label['frame'])
            pred_path: str = str(os.path.join(save_path, dirname + '_' + os.path.basename(label['path']) + '.pred.png'))
            save_pianoroll(pred_path, prediction['onset'], prediction['frame'])
            midi_path: str = str(os.path.join(save_path, dirname + '_' + os.path.basename(label['path']) + '.pred.mid'))
            save_midi(midi_path, p_est, i_est, v_est)

    return metrics


def evaluate_file(model_file_or_dir: str, piano_roll_audio_dataset_name: str, dataset_group: str, sequence_length: int,
                  save_path: str, onset_threshold: float, frame_threshold: float, device: str):
    piano_roll_audio_dataset = determine_datasets(piano_roll_audio_dataset_name, dataset_group, sequence_length, device)

    model = torch.load(model_file_or_dir, map_location=device).eval()
    summary(model)

    metrics: dict = evaluate(piano_roll_audio_dataset, model, onset_threshold, frame_threshold, save_path)

    total_eval_str: str = ''
    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            eval_str: str = f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}'
            print(eval_str)
            logging.info(eval_str)
            total_eval_str += eval_str + '\n'

    if save_path is not None:
        metrics_filepath = os.path.join(save_path, f'metrics-{piano_roll_audio_dataset_name}.txt')
        with open(metrics_filepath, 'w') as f:
            f.write(total_eval_str)


def evaluate_dir(model_dir: str, piano_roll_audio_dataset_name: str, dataset_group: str, sequence_length: int,
                 save_path: str, device: str):
    piano_roll_audio_dataset = determine_datasets(piano_roll_audio_dataset_name, dataset_group, sequence_length, device)


def determine_datasets(piano_roll_audio_dataset_name, dataset_group, sequence_length, device) \
        -> dataset_module.PianoRollAudioDataset:
    dataset_groups: List[str] = dataset_group.split(',')
    dataset_class = getattr(dataset_module, piano_roll_audio_dataset_name)
    kwargs = {'sequence_length': sequence_length, 'device': device}
    if dataset_group is not None:
        kwargs['groups'] = dataset_groups
    piano_roll_audio_dataset: dataset_module.PianoRollAudioDataset = dataset_class(**kwargs)
    return piano_roll_audio_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file_or_dir', type=str)
    parser.add_argument('piano_roll_audio_dataset_name', nargs='?', default='MAPS')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    save_path_arg = parser.parse_args().save_path
    if save_path_arg is not None:
        if os.path.exists(save_path_arg) and len(os.listdir(save_path_arg)) > 0:
            logging.warning(f'save_path {save_path_arg} is not empty. Clearing directory!')
            shutil.rmtree(save_path_arg)
        elif not os.path.exists(save_path_arg):
            os.makedirs(save_path_arg)

    dataset_name: str = parser.parse_args().piano_roll_audio_dataset_name
    datetime_str: str = datetime.now().strftime('%y%m%d-%H%M')
    logging_filepath = os.path.join('runs', f'evaluation-{dataset_name}-{datetime_str}.log')
    # filemode=a -> append
    logging.basicConfig(filename=logging_filepath, filemode="a", level=logging.INFO)
    if not os.path.exists(logging_filepath):
        raise Exception('logging file was not created!')

    model_file_or_dir_local: str = parser.parse_args().model_file_or_dir
    if os.path.isdir(model_file_or_dir_local):
        args: argparse.Namespace = parser.parse_args()
        if args.onset_threshold != 0.5 or args.frame_threshold != 0.5:
            raise ValueError(
                f'Explicitely set onset_threshold: {args.onset_threshold}, frame_threshold: {args.frame_threshold} '
                f'to value different to 0.5. This is not supported when already finished transcriptions are evaluated.')

    elif os.path.isfile(model_file_or_dir_local):
        with torch.no_grad():
            evaluate_file(**vars(parser.parse_args()))
    else:
        raise RuntimeError(f'model_file_or_dir {model_file_or_dir_local} does not exist!')
