import json
import logging
import os
import re
from abc import abstractmethod
from glob import glob
from typing import List, Tuple

import librosa
import numpy as np
import soundfile
import pandas as pd

from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .midi import parse_midi, save_np_arr_as_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            input_files: Tuple
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                # the asterisk means that the data is unpacked (Tuple is unpacked into function arguments)
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio: np.ndarray
        audio, sr = soundfile.read(audio_path, dtype='int16')
        # Conversion to fload see:
        # https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
        audio = np.array(audio).astype(np.float32)
        if sr != SAMPLE_RATE:
            logging.debug(f"Sample Rate Mismatch: resampling file: {audio_path}")
            audio = librosa.resample(audio, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group) -> List[Tuple]:
        """
        Args:
            group: e.g. train, validation, test
        Returns: the list of input files (audio_filename, tsv_filename) for this group
        to my understanding, the tsv contains onset, offset, note value and velocity of each note
        """
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v1.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if
                            row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in
                     files]

        result = []
        audio_path: str
        midi_path: str
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi: np.ndarray = parse_midi(midi_path)
                # midi is an array consisting of onset, offset, note and velocity
                # See other explanation on np.savetxt
                # noinspection PyTypeChecker
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed,
                         device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl',
                'StbgTGd2']

    def files(self, group) -> List[Tuple]:
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]

        assert (all(os.path.isfile(flac) for flac in flacs))
        assert (all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))


class SchubertWinterreiseDataset(PianoRollAudioDataset):
    r"""
    Goal: train Onsets and Frames on Schubert Winterreise data:
    Issues: We don't have one midi file assigned to a raw audio. We have a general score level midi file
            Therefore, we can't just use the midi file and the raw audio for training.
    Further Information:
            We have different types of measurements, especially the ann_audio_measure measurement
            Therefore, we can align these measurements with the midi -> warping the midi to match the measurement
    Solution:   Using the annotations to match midi with the raw data
                e.g. use ann_audio_measure -> Warp each measure
                another option: ann_audio_structure
    """

    def __init__(self,
                 path='data/Schubert_Winterreise_Dataset_v2-1', groups=None, sequence_length=None, seed=42,
                 device=DEFAULT_DEVICE):
        super().__init__(path,
                         groups if groups is not None else ['AL98', 'FI55', 'FI66', 'FI80', 'OL06', 'QU98', 'TR99'],
                         sequence_length, seed, device)

    @staticmethod
    def get_filenames_from_group(directory: str, regex_pattern) -> List[str]:
        files = glob(os.path.join(directory, '*.wav'))
        matching_files = [file for file in files if re.compile(fr".*{re.escape(regex_pattern)}.*").search(file)]
        return matching_files

    @staticmethod
    def combine_audio_midi(audio_filenames: List[str], midi_filenames: List[str]) -> List[Tuple]:
        """
        This is intended as the method which finally does the warping of the Schubert Midi files
        Args:
            audio_filenames: List of all audio filenames
            midi_filenames: List of all midi filenames
        Returns: audio - midi filename combination in the form of a List of tuples
        """
        audio_midi_combination: List[Tuple] = []
        for audio_filename in audio_filenames:
            basename = os.path.basename(audio_filename)
            # get number of piece
            number_str: str = basename[14:16]
            # Find matching midi file
            matching_files = [midi_file for midi_file in midi_filenames if
                              re.compile(fr".*-{number_str}.*").search(midi_file)]
            if len(matching_files) > 1:
                raise RuntimeError(f"Found more than one matching file for audio filename: {audio_filename}")
            midi_filepath: str = matching_files[0]
            # Create tuple
            audio_midi_combination.append((os.path.basename(audio_filename), os.path.basename(midi_filepath)))
        return audio_midi_combination

    @classmethod
    def available_groups(cls) -> List[str]:
        r"""
        HU33, SC06 are the public datasets -> these are used preferred for testing
        Returns: Available groups
        """
        return ['AL98', 'FI55', 'FI66', 'FI80', 'HU33', 'OL06', 'QU98', 'SC06', 'TR99']

    def files(self, group: str) -> List[Tuple]:
        """
        Args:
            group: group to return the filenames for. See self.available_groups() for the groups
        Returns:
        """
        audio_filenames: List[str] = []
        # This is List of Tuples containing the midi/audio combination for each file

        # for each group get the files
        # todo adding some form of handling wav and flac filenames
        audio_filenames.extend(
            self.get_filenames_from_group(os.path.join(self.path, '01_RawData', 'audio_wav'), group))

        midi_filenames: List[str] = glob(os.path.join(self.path, '01_RawData', 'score_midi', '*.mid'))

        files_audio_midi: List[Tuple] = self.combine_audio_midi(audio_filenames, midi_filenames)

        ann_audio_globalkey: pd.DataFrame = pd.read_csv(
            os.path.join(self.path, '02_Annotations', 'ann_audio_globalkey.csv'), sep=';')

        result: List[Tuple] = []
        audio_filename: str
        midi_filename: str
        tsv_dir: str = os.path.join(self.path, '01_RawData', 'score_tsv')
        if not os.path.exists(tsv_dir):
            os.makedirs(tsv_dir)
        for audio_filename, midi_filename in files_audio_midi:
            tsv_filename = audio_filename.replace('.mid', '.tsv').replace('.wav', '.tsv')
            if not os.path.exists(os.path.join(tsv_dir, tsv_filename)):
                self.create_tsv(ann_audio_globalkey, audio_filename, midi_filename, os.path.join(tsv_dir, tsv_filename))
            result.append((os.path.join(self.path, '01_RawData', 'audio_wav', audio_filename),
                           os.path.join(tsv_dir, tsv_filename)))
        return result

    def create_tsv(self, ann_audio_globalkey, audio_filename, midi_filename, tsv_filepath):
        work_id: str = audio_filename[:16]
        performance_id: str = audio_filename[17:21]
        column: pd.DataFrame = ann_audio_globalkey[(ann_audio_globalkey['WorkID'] == work_id) & (
                ann_audio_globalkey['PerformanceID'] == performance_id)]
        if len(column) != 1:
            raise RuntimeError(
                "Didn't find the matching annotion for global key offset. Please check manually.")
        global_key_offset: int = -column['transposeToMatchScore'].item()
        logging.info(
            f'Parsing midi file: {os.path.basename(midi_filename)} for audio {os.path.basename(audio_filename)} '
            f'with offset {str(global_key_offset)}')
        midi: np.ndarray = parse_midi(str(os.path.join(self.path, '01_RawData', 'score_midi', midi_filename)),
                                      global_key_offset)
        # This is for debugging the tsv creation process -> You can listen to the midi afterwards
        # save_np_arr_as_midi(midi, str(os.path.join(os.path.dirname(tsv_filepath), audio_filename + '.mid')))
        # For some reason pycharm expects an int value in np.savetxt() midi is ofc not an int value.
        # But this error is from pycharm. Therefore, the inspection is disabled here.
        # noinspection PyTypeChecker
        np.savetxt(tsv_filepath, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')


class SchubertWinterreisePiano(PianoRollAudioDataset):
    ...


class SchubertWinterreiseVoice(PianoRollAudioDataset):
    ...
