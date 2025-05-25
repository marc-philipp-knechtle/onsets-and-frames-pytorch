import argparse
import os
import re
import shutil
import sys
import logging
from collections import defaultdict
from datetime import datetime
from glob import glob
from typing import List, Dict, Tuple

import numpy as np
import pretty_midi
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import IterableDataset
import onsets_and_frames.dataset as dataset_module
from onsets_and_frames import *
from onsets_and_frames.dataset import PianoRollAudioDataset, SchubertWinterreiseDataset, SchubertWinterreisePiano, \
    SchubertWinterreiseVoice
from onsets_and_frames.decoding import extract_notes_from_frames
from onsets_and_frames.midi import parse_midi, create_midi
from onsets_and_frames.utils import save_pianoroll_matplotlib

from typing import List, Tuple

import scipy.integrate as sc_integrate
from sklearn.metrics import auc

import matplotlib.pyplot as plt

eps = sys.float_info.epsilon

default_evaluation_datasets: List[Tuple[str, dataset_module.PianoRollAudioDataset]] = \
    [
        # ('runs/MAESTRO', MAESTRO(groups=['test'])),
        # ('runs/SchubertWinterreiseDataset', SchubertWinterreiseDataset(groups=['SC06'])),
        # ('runs/SchubertWinterreisePiano', SchubertWinterreisePiano(groups=['SC06'])),
        # ('runs/SchubertWinterreiseVoice', SchubertWinterreiseVoice(groups=['SC06']))
    ]


def evaluate(pianoroll_dataset: IterableDataset, model: OnsetsAndFrames, onset_threshold=0.5,
             frame_threshold=0.5, save_path=None) -> dict:
    metrics = defaultdict(list)
    # Tensor for:
    # onset, offset, frame and velocity
    prediction: Dict[Tensor, Tensor, Tensor, Tensor]
    for label in tqdm(pianoroll_dataset):
        losses: Dict[str, Tensor]
        """
        includes:
        loss/onset
        loss/offset
        loss/frame
        loss/velocity
        """
        prediction, losses = model.run_on_batch(label)

        loss: Tensor
        for key, loss in losses.items():
            metrics[key].append(loss.item())

        # key are: onset, offset, frame and velocity
        # values are torch tensors
        # onset shape: [frame_indices, 88]
        # offset shape: [frame_indices, 88]
        # frame shape: [frame_indices, 88]
        # velocity shape: [frame_indices, 88]
        for key, value in prediction.items():
            # apply relu = sets all negative values to 0
            # .squeeze(0) = remove the first dimension if it's size one
            # I'm not entirely sure why the .squeeze(0) function is necessary, however it does not do any harm!
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
        # t_ref = np.ndarray containing the frame indices
        t_ref: np.ndarray
        # list of np.ndarray, each containing the frequency bin indices
        f_ref: List[np.ndarray]
        p_ref, i_ref, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])
        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)
        """
        estimate = prediction
        pitches 
        intervals
        velocities
        """

        required_keys_oaf = ['onset', 'frame', 'velocity']
        required_keys_frame_model = ['frame']

        if all(key in prediction for key in required_keys_oaf):
            p_est, i_est, v_est = extract_notes(prediction['onset'], prediction['frame'], prediction['velocity'],
                                                onset_threshold, frame_threshold)
        elif all(key in prediction for key in required_keys_frame_model):
            p_est, i_est, v_est = extract_notes_from_frames(prediction['frame'], frame_threshold)
        else:
            raise RuntimeError(f'Expected keys {required_keys_oaf} or {required_keys_frame_model} in prediction.')

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

        # Commenting the velocity related labels here because we train wout velocity
        # p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
        #                                           offset_ratio=None, velocity_tolerance=0.1)
        # metrics['metric/note-with-velocity/precision'].append(p)
        # metrics['metric/note-with-velocity/recall'].append(r)
        # metrics['metric/note-with-velocity/f1'].append(f)
        # metrics['metric/note-with-velocity/overlap'].append(o)
        #
        # p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
        # metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        # metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        # metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        # metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)
        metrics['metric/frame/f1'].append(
            hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

            # dirname is included in the save_path stored evaluation files to provide more context
            # e.g. performance year for MAESTRO
            # or vocal or accompaniment for SWD Vocal/Piano
            dirname: str = os.path.basename(os.path.dirname(label['path']))
            label_path: str = str(os.path.join(save_path, dirname + '_' + os.path.basename(label['path'])))

            # saving pianorolls of label
            # save_pianoroll(label_path + '.label.png', label['onset'], label['frame'])
            midifile_reference: pretty_midi.PrettyMIDI = create_midi(i_ref, p_ref, v_ref)
            save_pianoroll_matplotlib(midifile_reference, label_path + '.ref.png', start_param=10, end_param=25)

            # saving pianorolls of prediction
            # save_pianoroll(label_path + '.pred.png', prediction['onset'], prediction['frame'])
            midi_path: str = str(os.path.join(save_path, dirname + '_' + os.path.basename(label['path']) + '.pred.mid'))
            midifile_prediction: pretty_midi.PrettyMIDI = save_midi(midi_path, p_est, i_est, v_est)
            save_pianoroll_matplotlib(midifile_prediction, label_path + '.pred.png', start_param=10, end_param=25,
                                      onset_prediction=prediction['onset'])

    return metrics


def evaluate_ap(pianoroll_dataset: IterableDataset, model: OnsetsAndFrames) -> List[float]:
    """

    Args:
        pianoroll_dataset:
        model:

    Returns:

    """
    prediction: Dict[Tensor, Tensor, Tensor, Tensor]
    average_precisions: List[float] = []
    for label in tqdm(pianoroll_dataset):
        prediction, losses = model.run_on_batch(label)

        for key, value in prediction.items():
            # apply relu = sets all negative values to 0
            # .squeeze(0) = remove the first dimension if it's size one
            # I'm not entirely sure why the .squeeze(0) function is necessary, however it does not do any harm!
            value.squeeze_(0).relu_()

        # list of np.ndarray, each containing the frequency bin indices
        f_ref: List[np.ndarray]
        p_ref, i_ref, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])
        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in freqs]) for freqs in f_ref]

        precision_recall_pairs_frame: List[Tuple[float, float]] = []

        for threshold in np.arange(0, 1.0, 0.05):
            p_est, i_est, v_est = extract_notes(prediction['onset'], prediction['frame'], prediction['velocity'],
                                                threshold, threshold)
            t_est, f_est = notes_to_frames(p_est, i_est, prediction['frame'].shape)
            t_est = t_est.astype(np.float64) * scaling
            f_est = [np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in freqs]) for freqs in f_est]

            frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)

            precision_recall_pairs_frame.append((frame_metrics['Precision'], frame_metrics['Recall']))

        ap = calc_ap_from_prec_recall_pairs(precision_recall_pairs_frame, plot=False,
                                            thresholds=np.arange(0, 1.0, 0.05).tolist(), title=f"{label['path']}")
        average_precisions.append(ap)

    return average_precisions


def calc_ap_from_prec_recall_pairs(precision_recall_pairs: List[Tuple[float, float]], plot: bool,
                                   thresholds, title: str) -> float:
    """
    Calculate AP with sklearn.metrics.auc (Area Under Curve)
    This is an alternative method to sklearn.metrics.average_precision_score
    We cannot use said method in some cases because we rely on multiple thresholds (e.g. onset thresh and frame thresh)
    Args:
        precision_recall_pairs: List of Tuples with (precision, recall) values.
        plot: ...
        thresholds: ...
        title: ...
    Returns: Average Precision score, calculated with sklearn
    """
    precision_recall_pairs_sorted_precision = sorted(precision_recall_pairs, key=lambda pair: pair[0])
    precision, recall = zip(*precision_recall_pairs_sorted_precision)
    ap = auc(precision, recall)
    if plot:
        plot_rec_rec_curve(precision, recall, thresholds, title)
    assert ap >= 0
    return ap


def plot_rec_rec_curve(precision: List[float], recall: List[float], thresholds: List[float] = None, title: str = None):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(recall, precision)
    if thresholds is not None:
        thresholds = tuple(thresholds)
        for x, y, thr in zip(recall, precision, thresholds):
            ax.annotate(f"({thr:.2f})", (x, y), textcoords="offset points", xytext=(5, 5), ha='center')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title if title is not None else 'Precision-Recall Curve')
    plt.show()


def compute_avg_precision(piano_roll_audio_dataset: PianoRollAudioDataset, model: OnsetsAndFrames) -> Tuple[
    List[float], List[float]]:
    """
    Args:
        piano_roll_audio_dataset:
        model:

    Returns: Average Precision (see paper explanation) for each evaluated file (therefore as a list)
    """
    logging.info('Computing average Precision Recall (Area under Precision, Recall curve)')
    scaling = HOP_LENGTH / SAMPLE_RATE

    avg_precision_frame: List[float] = []
    avg_precision_onset: List[float] = []
    for label in tqdm(piano_roll_audio_dataset):
        prediction, losses = model.run_on_batch(label)

        for key, value in prediction.items():
            value.squeeze_(0).relu_()

        p_ref, i_ref, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])
        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)

        i_ref_scaled_reshaped = (i_ref * scaling).reshape(-1, 2)
        p_ref_hz = np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in p_ref])
        t_ref_scaled = t_ref.astype(np.float64) * scaling
        f_ref_hz = [np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in freqs]) for freqs in f_ref]

        precision_recall_pairs_frame: List[Tuple[float, float]] = []
        precision_recall_pairs_onset: List[Tuple[float, float]] = []
        for threshold in np.arange(0, 1.1, 0.05):
            p_est, i_est, v_est = extract_notes(prediction['onset'], prediction['frame'], prediction['velocity'],
                                                threshold, threshold)
            t_est, f_est = notes_to_frames(p_est, i_est, prediction['frame'].shape)

            i_est_scaled_reshaped = (i_est * scaling).reshape(-1, 2)
            p_est_hz = np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in p_est])
            t_est_scaled = t_est.astype(np.float64) * scaling
            f_est_hz = [np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in freqs]) for freqs in f_est]

            frame_metrics: Dict[str, float] = evaluate_frames(t_ref_scaled, f_ref_hz, t_est_scaled, f_est_hz)
            precision: float = frame_metrics['Precision']
            recall: float = frame_metrics['Recall']
            precision_recall_pairs_frame.append((recall, precision))

            p_onset, r_onset, f, o = evaluate_notes(i_ref_scaled_reshaped, p_ref_hz, i_est_scaled_reshaped, p_est_hz,
                                                    offset_ratio=None)
            precision_recall_pairs_onset.append((p_onset, r_onset))

        precision_recall_pairs_frame = sorted(precision_recall_pairs_frame)

        total_precision_recall_area = 0
        prev_recall = 0
        prev_precision = 0
        for recall, precision in precision_recall_pairs_frame:
            total_precision_recall_area += (recall - prev_recall) * max(prev_precision, precision)
            prev_precision = precision
            prev_recall = recall
        avg_precision_frame.append(total_precision_recall_area)

        total_precision_recall_area = 0
        prev_recall = 0
        prev_precision = 0
        for recall, precision in precision_recall_pairs_onset:
            total_precision_recall_area += (recall - prev_recall) * max(prev_precision, precision)
            prev_precision = precision
            prev_recall = recall
        avg_precision_onset.append(total_precision_recall_area)

    return avg_precision_frame, avg_precision_onset


def evaluate_model_file(model_file: str, piano_roll_audio_dataset_name: str, dataset_group: str, sequence_length: int,
                        save_path: str, onset_threshold: float, frame_threshold: float, device: str):
    piano_roll_audio_dataset = determine_datasets(piano_roll_audio_dataset_name, dataset_group, sequence_length, device)

    model: OnsetsAndFrames = torch.load(model_file, map_location=device).eval()
    summary(model)

    evaluate_model(model, piano_roll_audio_dataset, frame_threshold, onset_threshold, save_path)


def evaluate_model(model: OnsetsAndFrames, piano_roll_audio_dataset: PianoRollAudioDataset, frame_threshold: float,
                   onset_threshold: float, save_path: str):
    metrics: Dict[str, List[float]] = evaluate(piano_roll_audio_dataset, model, onset_threshold, frame_threshold,
                                               save_path)
    avg_precision_frame, avg_precision_onset = compute_avg_precision(piano_roll_audio_dataset, model)
    metrics.update({'metric/frame/avg_precision': avg_precision_frame})
    metrics.update({'metric/note/avg_precision_onset': avg_precision_onset})
    """
    evaluate returns a list of metrics which are included by file level (meaning each metric compute result is stored)
    key: metric name
    value(s): list off all computations
    """
    total_eval_str: str = ''
    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            eval_str: str = f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}'
            print(eval_str)
            logging.info(eval_str)
            total_eval_str += eval_str + '\n'
    if save_path is not None:
        metrics_filepath = os.path.join(save_path, f'metrics-{str(piano_roll_audio_dataset)}.txt')
        with open(metrics_filepath, 'w') as f:
            f.write(total_eval_str)


def evaluate_inference_dir(model_dir: str, piano_roll_audio_dataset_name: str, dataset_group: str, sequence_length: int,
                           save_path: str, device: str):
    piano_roll_audio_dataset = determine_datasets(piano_roll_audio_dataset_name, dataset_group, sequence_length, device)

    metrics = defaultdict(list)

    predictions_filepaths: List[str] = glob(os.path.join(model_dir, '*.mid'))

    label: Dict
    for label in tqdm(piano_roll_audio_dataset):
        # This only works for the vocano annotations as they represent the directory structure created by spleeter
        dirname: str = os.path.dirname(label['path'])
        label_name = os.path.basename(label['path'])
        label_name = label_name.replace('.wav', '')
        matching_midi_filepaths = [csv_file for csv_file in predictions_filepaths if
                                   re.compile(fr".*{re.escape(label_name)}.*").search(csv_file)]
        if len(matching_midi_filepaths) != 1:
            raise RuntimeError(
                f'Found different amount of predictions for label {label_name}. '
                f'Expected 1, found {len(matching_midi_filepaths)}')
        midi_filepath: str = matching_midi_filepaths[0]
        midifile: np.ndarray = parse_midi(midi_filepath)

        pitches = []
        intervals = []
        velocities = []
        for start_time, end_time, pitch, velocity in midifile:
            pitches.append(int(pitch))
            intervals.append([start_time, end_time])
            velocities.append(velocity)

        scaling = HOP_LENGTH / SAMPLE_RATE

        # Array of estimated pitch values in Hertz (later converted from midi to hz)
        p_est: np.ndarray = np.array(pitches)
        p_est_hz = np.array([midi_to_hz(midi_val) for midi_val in p_est])
        # Array of estimated notes time intervals (onset and offset times)
        i_est: np.ndarray = np.array(intervals)
        v_est: np.ndarray = np.array(velocities)

        i_est_frames = (i_est * SAMPLE_RATE / HOP_LENGTH).reshape(-1, 2).astype(int)

        """
        reference = ground truth value
        pitches
        intervals
        velocities
        """
        p_ref: np.ndarray
        # Array of reference notes time intervals(onset and offset times)
        i_ref_frames: np.ndarray
        v_ref: np.ndarray

        p_ref, i_ref_frames, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])

        # scaling = HOP_LENGTH/SAMPLE_RATE = Conversion from time bin indices back to realtime
        # .reshape(-1, 2) has no effect as far as my debugging is concerned
        # It would convert the array such that it consists of x length in the first dimension and length 2 in the second
        # (because of interval format)
        # however, it naturally has this format.
        i_ref = (i_ref_frames * scaling).reshape(-1, 2)
        p_ref_hz = np.array([midi_to_hz(MIN_MIDI + midi_val) for midi_val in p_ref])

        p, r, f, o = evaluate_notes(i_ref, p_ref_hz, i_est, p_est_hz, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref_hz, i_est, p_est_hz)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref_hz, v_ref, i_est, p_est_hz, v_est,
                                                  offset_ratio=None, velocity_tolerance=0.1)
        metrics['metric/note-with-velocity/precision'].append(p)
        metrics['metric/note-with-velocity/recall'].append(r)
        metrics['metric/note-with-velocity/f1'].append(f)
        metrics['metric/note-with-velocity/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref_hz, v_ref, i_est, p_est_hz, v_est,
                                                  velocity_tolerance=0.1)
        metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

        # this calculates the frame based metrics
        # this is only useful when having a nuanced dataset
        p_ref_min_midi = np.array([x + MIN_MIDI for x in p_ref])
        # t_ref = np.ndarray containing the frame indices
        t_ref, f_ref = notes_to_frames(p_ref_min_midi, i_ref_frames, label['frame'].shape)
        t_est, f_est = notes_to_frames(p_est, i_est_frames, label['frame'].shape)
        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)
        metrics['metric/frame/f1'].append(
            hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

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
    parser.add_argument('piano_roll_audio_dataset_name', nargs='?', default='default',
                        help='PianoRollAudioDataset where the model is evaluated on.'
                             'Default is default which evaluates on every configured dataset.')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args: argparse.Namespace = parser.parse_args()

    dataset_name: str = parser.parse_args().piano_roll_audio_dataset_name
    datetime_str: str = datetime.now().strftime('%y%m%d-%H%M')
    logging_filepath: str
    if args.save_path is None:
        logging_filepath = os.path.join('runs', f'evaluation-{datetime_str}.log')
    else:
        logging_filepath = os.path.join(args.save_path, f'evaluation-{dataset_name}-{datetime_str}.log')
    if not os.path.exists(os.path.dirname(logging_filepath)):
        os.makedirs(os.path.dirname(logging_filepath))
    # filemode=a -> append
    logging.basicConfig(filename=logging_filepath, filemode="a", level=logging.INFO)

    # By default, the root logger is set to WARNING and all loggers you define
    # inherit that value. Here we set the root logger to NOTSET. This logging
    # level is automatically inherited by all existing and new sub-loggers
    # that do not set a less verbose level.
    logging.root.setLevel(logging.NOTSET)

    if not os.path.exists(logging_filepath):
        raise Exception('logging file was not created!')

    save_path_arg = parser.parse_args().save_path
    # todo check with os.makedirs above
    # todo maybe raise runtime error if exists and is not empty!
    # if save_path_arg is not None:
    #     if os.path.exists(save_path_arg) and len(os.listdir(save_path_arg)) > 0:
    #         logging.warning(f'save_path {save_path_arg} is not empty. Clearing directory!')
    #         shutil.rmtree(save_path_arg)
    #     elif not os.path.exists(save_path_arg):
    #         os.makedirs(save_path_arg)

    model_file_or_dir_local: str = parser.parse_args().model_file_or_dir
    if os.path.isdir(model_file_or_dir_local):  # = if we evaluate a directory with already created annotations
        if args.onset_threshold != 0.5 or args.frame_threshold != 0.5:
            raise ValueError(
                f'Explicitely set onset_threshold: {args.onset_threshold}, frame_threshold: {args.frame_threshold} '
                f'to value different to 0.5. This is not supported when already finished transcriptions are evaluated.')
        evaluate_inference_dir(args.model_file_or_dir, args.piano_roll_audio_dataset_name, args.dataset_group,
                               args.sequence_length, args.save_path, args.device)
    elif os.path.isfile(model_file_or_dir_local):
        if args.piano_roll_audio_dataset_name == 'default':
            if args.dataset_group is not None:
                raise RuntimeError('Specified group with default evaluation. Default runs all possible evaluations.'
                                   'You cannot specify a fixed group for this kind of evaluation.')
            if args.save_path is not None:
                raise RuntimeError('Specified save path with default evaluation. Default evaluation has specified save'
                                   'paths. Please use the default ones or correct them via code.')
            for save_path_key, dataset_item in default_evaluation_datasets:
                logging.info(f'Evaluating on {str(dataset_item)} with saving at: {save_path_key}')
                print(f'Evaluating on {str(dataset_item)} with saving at: {save_path_key}')
                with torch.no_grad():
                    model_onsets: OnsetsAndFrames = torch.load(args.model_file_or_dir, map_location=args.device).eval()
                    summary(model_onsets)
                    evaluate_model(model_onsets, dataset_item, args.frame_threshold, args.onset_threshold,
                                   save_path_key)
        else:
            with torch.no_grad():
                evaluate_model_file(args.model_file_or_dir, args.piano_roll_audio_dataset_name, args.dataset_group,
                                    args.sequence_length, args.save_path, args.onset_threshold, args.frame_threshold,
                                    args.device)
    else:
        raise RuntimeError(f'model_file_or_dir {model_file_or_dir_local} does not exist!')
