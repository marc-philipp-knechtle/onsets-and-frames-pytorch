import json
import logging
import os
import re
import shutil
import pretty_midi

from abc import abstractmethod
from glob import glob
from typing import List, Tuple, Dict

import librosa
import numpy as np
import pandas as pd
import soundfile
import torch
from torch import Tensor

from torch.utils.data import Dataset
from tqdm import tqdm

import transcribe
from . import midi
from .constants import *
from .midi import parse_midi

dataset_definitions = {
    'maestro_training': lambda: MAESTRO(groups=['train'], sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'maestro_validation': lambda: MAESTRO(groups=['validation'], sequence_length=DEFAULT_SEQUENCE_LENGTH),

    'winterreise_training': lambda: SchubertWinterreiseDataset(groups=['FI66', 'FI80', 'OL06', 'QU98', 'TR99'],
                                                               sequence_length=DEFAULT_SEQUENCE_LENGTH,
                                                               neither_split='train'),
    'winterreise_validation': lambda: SchubertWinterreiseDataset(groups=['AL98', 'FI55'],
                                                                 sequence_length=DEFAULT_SEQUENCE_LENGTH,
                                                                 neither_split='validation'),
    'winterreisevoice_training': lambda: SchubertWinterreiseVoice(
        groups=['FI55', 'FI66', 'FI80', 'OL06', 'QU98', 'TR99'],
        sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'winterreisevoice_validation': lambda: SchubertWinterreiseVoice(groups=['AL98'],
                                                                    sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'winterreisepiano_training': lambda: SchubertWinterreisePiano(
        groups=['FI55', 'FI66', 'FI80', 'OL06', 'QU98', 'TR99'],
        sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'winterreisepiano_validation': lambda: SchubertWinterreisePiano(groups=['AL98'],
                                                                    sequence_length=DEFAULT_SEQUENCE_LENGTH),

    'maps_training': lambda: MAPS(
        groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
        sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'maps_validation': lambda: MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=DEFAULT_SEQUENCE_LENGTH),

    # Furtwangler1953,KeilberthFurtw1952,Krauss1953
    'wrd_test': lambda: WagnerRingDataset(groups=['Furtwangler1953', 'KeilberthFurtw1952', 'Krauss1953'],
                                          sequence_length=DEFAULT_SEQUENCE_LENGTH),

    'b10_train': lambda: Bach10Dataset(groups=['01', '02', '03', '04']),
    'b10_validation': lambda: Bach10Dataset(groups=['05', '06']),

    'PhA_train': lambda: PhenicxAnechoicDataset(groups=['beethoven', 'mahler']),

    'CSD_train': lambda: ChoralSingingDataset(groups=['Traditional_ElRossinyol']),
    'CSD_validation': lambda: ChoralSingingDataset(groups=['Guerrero_NinoDios']),

    'MuN_train': lambda: MusicNetDataset(groups=['MuN-10-var-train']),
    'MuN_validation': lambda: MusicNetDataset(groups=['MuN-validation']),
    'MuN_test': lambda: MusicNetDataset(groups=['MuN-10-var-test']),
    'MuN_non-piano_train': lambda: MusicNetDataset(groups=['MuN-non-piano-tr']),
    'MuN_non-piano_validation': lambda: MusicNetDataset(groups=['MuN-non-piano-val']),

    'RWC_non-piano_train': lambda: RwcDataset(groups=['non-piano-train']),
    'RWC_non-piano_validation': lambda: RwcDataset(groups=['non-piano-validation'])
}


class PianoRollAudioDataset(Dataset):
    path: str
    groups: List[str]
    sequence_length: int
    device: str
    random: np.random.RandomState
    data: List[Dict]

    def __init__(self, path, groups=None, sequence_length=DEFAULT_SEQUENCE_LENGTH, seed=42, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.data: List[Dict[str, Tensor]] = []

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
            # audio_length_seconds = audio_length / SAMPLE_RATE
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin_sample = step_begin * HOP_LENGTH
            end_sample = begin_sample + self.sequence_length

            result['audio'] = data['audio'][begin_sample:end_sample].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        """
        Calculating the length of one step to get the offset length in ground truth: 
        see constants.py Onset Length
        Sample Rate = 16000
        Hop Length = 16000 * 32 // 1000 = 512
        Onset Length s = 512 / 16000 = 0.032
        """
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def clear_computed(self):
        logging.info("Clearing .pt files created by PianoRollAudioDataset.load().\n"
                     "This is because clear_computed is set as true.\n"
                     "The .pt files are created again for this run. For a faster execution you have to disable "
                     "clear_computed.")
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pt"):
                    os.remove(os.path.join(root, file))

    @staticmethod
    def load(audio_path: str, tsv_path: str) -> Dict[str, Tensor]:
        """
        load an audio track and the corresponding labels (tsv) annotations

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
            try:
                return torch.load(saved_data_path)
            except EOFError:
                print(f'File {saved_data_path} is corrupted. Please inspect manually, or delete.')

        """
        # Previous audio loading code. This did not work because of several issues:
        #     * The default handling with audio, sr = soundfile.read(audio_path, dtype='int16') did not work with MuN
        #     * Changing this to float32 dtype worked for loading the audio file, but did not produce correct inference
        audio: np.ndarray
        audio, sr = soundfile.read(audio_path, dtype='float32') # Previously: dtype='int16' (did no work bc of MuN)
        # Conversion to fload see:
        # https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
        audio: np.ndarray = np.array(audio).astype(np.float32)
        if len(audio.shape) > 1:
            # Convert Stereo to Mono
            logging.warning('Audio is two-dimensional - this is a stereo file! Converting to mono!')
            audio = audio.T
            audio = audio[0]
        if sr != SAMPLE_RATE:
            logging.info(f"Sample Rate Mismatch: resampling file: {audio_path}")
            # This whole processing was required to handle stereo files. It's now commented because the issue is handled
            # above
            # if audio.shape[1] == 2:
            #     '''
            #     In some cases, there were issues with the format of audio (channels and floating point numbers were
            #     reversed. This happened when processing annotations generated by spleeter.
            #     Error Msg:
            #     This is causing ValueError: Input signal length=2 is too small to resample from x -> y
            #     This issue is also described here:
            #     https://github.com/librosa/librosa/issues/915
            #
            #     # todo check the shape of ordinary input
            #     '''
            #     logging.warning(f"Wrong formatted audio tuple. Transposing Tuple of file {audio_path}")
            #     audio = audio.T
            #     audio = librosa.resample(audio, sr, SAMPLE_RATE)
            #     audio = audio.T

            audio = librosa.resample(audio, sr, SAMPLE_RATE)
        """

        audio: np.ndarray
        sr: int
        # librosa is a wrapper for soundfile
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # audio normalization
        # see https://stackoverflow.com/questions/66066364/audio-volume-normalize-python
        # max_peak = np.max(np.abs(audio))
        # ratio = 1 / max_peak
        # audio = audio * ratio

        audio = transcribe.float_samples_to_int16(audio)

        assert sr == SAMPLE_RATE
        assert audio.dtype == 'int16'

        audio_tensor = torch.ShortTensor(audio)
        audio_length = len(audio_tensor)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        midi_data_from_tsv = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi_data_from_tsv:
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

        data: Dict[str, Tensor] = dict(path=audio_path, audio=audio_tensor, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device)

    def __str__(self):
        return 'MAESTRO'

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
            metadata_files = glob(os.path.join(self.path, '*.json'))
            if len(metadata_files) != 1:
                raise RuntimeError(
                    f'Unexpected number of metadata files found. Found {len(metadata_files)}, Expected 1')
            metadata = json.load(open(metadata_files[0]))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if
                            row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi_filename) for
                     audio, midi_filename in files]

        result = []
        audio_path: str
        midi_path: str
        for audio_path, midi_path in tqdm(files):
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi_arr: np.ndarray = parse_midi(midi_path)
                # midi_arr is an array consisting of onset, offset, note and velocity
                # See other explanation on np.savetxt
                # noinspection PyTypeChecker
                np.savetxt(tsv_filename, midi_arr, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed,
                         device)

    def __str__(self):
        return 'MAPS'

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
    Goal: train Onsets and Frames on Schubert Winterreise data
    """
    swd_midi: str
    swd_csv: str
    swd_tsv: str
    swd_audio_wav: str

    neither_split: str

    def __init__(self,
                 path='data/Schubert_Winterreise_Dataset_v2-1', groups=None, sequence_length=None, seed=42,
                 device=DEFAULT_DEVICE, neither_split=None):
        # adding underscore to symbolize that these annotations are computationally created
        self.swd_midi = os.path.join(path, '02_Annotations', '_ann_audio_note_midi')
        self.swd_csv = os.path.join(path, '02_Annotations', 'ann_audio_note')
        self.swd_tsv = os.path.join(path, '02_Annotations', '_ann_audio_note_tsv')
        self.swd_audio_wav = os.path.join(path, '01_RawData', 'audio_wav')

        self.neither_split = neither_split

        super().__init__(path, groups, sequence_length, seed, device)

    def __str__(self):
        return 'SchubertWinterreiseDataset'

    @staticmethod
    def get_filepaths_from_group(directory: str, regex_pattern) -> List[str]:
        files = glob(os.path.join(directory, '**', '*.wav'), recursive=True)
        matching_files = [file for file in files if re.compile(fr".*{re.escape(regex_pattern)}.*").search(file)]
        return matching_files

    @staticmethod
    def combine_audio_midi(audio_filenames: List[str], midi_filenames: List[str]) -> List[Tuple[str, str]]:
        """
        This is intended as the method which finally does the warping of the Schubert Midi files
        Args:
            audio_filenames: List of all audio filenames
            midi_filenames: List of all midi filenames
        Returns: audio - midi filename combination in the form of a List of tuples
        """
        audio_midi_combination: List[Tuple[str, str]] = []
        for audio_filename in audio_filenames:
            basename = os.path.basename(audio_filename)
            number_str: str = basename[14:16]
            performance: str = basename[17:21]
            # Find matching midi file
            matching_files = [midi_file for midi_file in midi_filenames if
                              re.compile(fr".*-{number_str}_{performance}.*").search(midi_file)]
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
        Returns: List[Tuple[audio_filepath, tsv_filepath]] is a List of all audio tsv file combinations for this piece
        """
        """
        Definition of the neither split: 
        Comparing: 
        
        For the version split, we use all songs in two versions for testing, two further versions for validation, 
        and the remaining five versions for training. For the song split, we use the songs 17–24 of Winterreise 
        in all versions for testing, songs 14–16 for validation, and songs 1–13 for training.
        
        testing: HU33, SC06, 17-24
        validation: AL98, FI55 14-16
        train: FI66, FI80, OL06, QU98, TR99 1-13
        """
        audio_filepaths: List[str] = sorted(self.get_filepaths_from_group(self.swd_audio_wav, group))
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        if self.neither_split is not None:
            if self.neither_split == 'train':
                audio_filepaths = audio_filepaths[:13]
            elif self.neither_split == 'validation':
                audio_filepaths = audio_filepaths[13:16]
            elif self.neither_split == 'test':
                audio_filepaths = audio_filepaths[16:25]

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.swd_csv, '*.csv'))
        # save csv as midi
        midi_path = midi.save_csv_as_midi(ann_audio_note_filepaths_csv, self.swd_midi)
        midi_audio_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))
        files_audio_audio_midi: List[Tuple[str, str]] = self.combine_audio_midi(audio_filepaths, midi_audio_filepaths)

        """
        The issue with this approach is that the midi_score_filenames are not individual transcriptions of the pieces. 
        However, they are just the score and each version is slightly different to the score! 
        """
        # midi_score_filenames: List[str] = glob(os.path.join(self.path, '01_RawData', 'score_midi', '*.mid'))
        # files_audio_score_midi: List[Tuple[str, str]] = self.combine_audio_midi(audio_filepaths, midi_score_filenames)
        # This method also isn't necessary anymore once the audio transcription works!
        # ann_audio_globalkey: pd.DataFrame = pd.read_csv(
        #     os.path.join(self.path, '02_Annotations', 'ann_audio_globalkey.csv'), sep=';')

        # This is List of Tuples containing the midi/audio combination for each file
        # convert midi into tsv
        result = self.create_audio_tsv(files_audio_audio_midi)
        return result

    def clear_computed(self):
        logging.info(f'Clearing dirs {self.swd_midi} and {self.swd_tsv}.\n'
                     f'This is because clear_computed is set to true. \n'
                     f'They are recomputed. For a faster execution, you have to disable clear_computed.')
        if os.path.exists(self.swd_midi):
            shutil.rmtree(self.swd_midi)
        if os.path.exists(self.swd_tsv):
            shutil.rmtree(self.swd_tsv)
        super().clear_computed()

    # todo refactor this like in SchubertWinterreiseVoice!
    def create_audio_tsv(self, files_audio_audio_midi: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Args:
            files_audio_audio_midi: List of all audio to midi combinations
        Returns: List[Tuple[audio_filepath, tsv_filepath]]
        """
        result: List[Tuple[str, str]] = []
        audio_filename: str
        midi_filename: str
        if not os.path.exists(self.swd_tsv):
            os.makedirs(self.swd_tsv)
        for audio_filename, midi_filename in files_audio_audio_midi:
            tsv_filename = audio_filename.replace('.mid', '.tsv').replace('.wav', '.tsv')
            if not os.path.exists(os.path.join(self.swd_tsv, tsv_filename)):
                midi.create_tsv_from_midi(os.path.join(self.swd_midi, midi_filename),
                                          os.path.join(self.swd_tsv, tsv_filename))
            result.append((os.path.join(self.swd_audio_wav, audio_filename), os.path.join(self.swd_tsv, tsv_filename)))
        return result


class SchubertWinterreisePiano(SchubertWinterreiseDataset):
    swd_piano_midi: str
    swd_piano_tsv: str
    swd_piano_wav: str
    swd_csv: str

    def __init__(self, path='data/Schubert_Winterreise_Dataset_v2-1', groups=None, sequence_length=None, seed=42,
                 device=DEFAULT_DEVICE):
        # adding underscores
        self.swd_piano_midi = os.path.join(path, '02_Annotations', '_ann_audio_piano_midi')
        self.swd_piano_tsv = os.path.join(path, '02_Annotations', '_ann_audio_piano_tsv')
        self.swd_piano_wav = os.path.join(path, '01_RawData', 'audio_wav_spleeter_separated')
        self.swd_csv = os.path.join(path, '02_Annotations', 'ann_audio_note')

        super().__init__(path,
                         groups if groups is not None else ['AL98', 'FI55', 'FI66', 'FI80', 'OL06', 'QU98', 'TR99'],
                         sequence_length, seed, device)

    def __str__(self):
        return 'SchubertWinterreisePiano'

    def create_audio_tsv(self, filepaths_audio_midi: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        audio_filepath: str
        midi_filepath: str
        if not os.path.exists(self.swd_piano_tsv):
            os.makedirs(self.swd_piano_tsv)
        for audio_filepath, midi_filepath in filepaths_audio_midi:
            tsv_filepath = os.path.join(self.swd_piano_tsv, os.path.basename(midi_filepath).replace('.mid', '.tsv'))
            if not os.path.exists(tsv_filepath):
                midi.create_tsv_from_midi(midi_filepath, tsv_filepath)
            result.append((audio_filepath, tsv_filepath))
        return result

    def files(self, group: str) -> List[Tuple]:
        audio_filepaths: List[str] = super().get_filepaths_from_group(self.swd_piano_wav, group)
        piano_audio_filepaths: List[str] = []
        for path in audio_filepaths:
            if path.__contains__('accompaniment'):
                piano_audio_filepaths.append(path)
        if len(piano_audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')
        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.swd_csv, '*.csv'))
        midi_path = midi.save_csv_as_midi(ann_audio_note_filepaths_csv, self.swd_piano_midi, instrument_arg='piano')
        midi_piano_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))
        files_piano_midi_filepaths: List[Tuple[str, str]] = SchubertWinterreiseVoice.combine_audio_midi(
            piano_audio_filepaths, midi_piano_filepaths)
        piano_tsv_filepaths = self.create_audio_tsv(files_piano_midi_filepaths)
        return piano_tsv_filepaths

    def clear_computed(self):
        logging.info(f'Clearing dirs {self.swd_piano_midi} and {self.swd_piano_tsv}.\n'
                     f'This is because clear_computed is set to true. \n'
                     f'They are recomputed. For a faster execution, you have to disable clear_computed.')
        if os.path.exists(self.swd_piano_midi):
            shutil.rmtree(self.swd_piano_midi)
        if os.path.exists(self.swd_piano_tsv):
            shutil.rmtree(self.swd_piano_tsv)
        super().clear_computed()


class SchubertWinterreiseVoice(SchubertWinterreiseDataset):
    swd_vocal_midi: str
    swd_vocal_tsv: str
    swd_vocal_wav: str
    swd_csv: str

    def __init__(self,
                 path='data/Schubert_Winterreise_Dataset_v2-1', groups=None, sequence_length=None, seed=42,
                 device=DEFAULT_DEVICE):
        # adding underscore to symbolize that these annotations are computationally created
        self.swd_vocal_wav = os.path.join(path, '01_RawData', 'audio_wav_spleeter_separated')
        self.swd_vocal_midi = os.path.join(path, '02_Annotations', '_ann_audio_voice_midi')
        self.swd_vocal_tsv = os.path.join(path, '02_Annotations', '_ann_audio_voice_tsv')
        self.swd_csv = os.path.join(path, '02_Annotations', 'ann_audio_note')
        super().__init__(path,
                         groups if groups is not None else ['AL98', 'FI55', 'FI66', 'FI80', 'OL06', 'QU98', 'TR99'],
                         sequence_length, seed, device)

    def __str__(self):
        return 'SchubertWinterreiseVoice'

    @staticmethod
    def combine_audio_midi(audio_filepaths: List[str], midi_filepaths: List[str]) -> List[Tuple[str, str]]:
        """
        The goal of this method is to assign the correct midi annotation to the audio version
        Args:
            audio_filepaths: audio filepaths (vocal)
            midi_filepaths: midi filepaths of all available midi annotations
        Returns: audio - midi filepath assignment
        """
        audio_midi_combination: List[Tuple[str, str]] = []
        for audio_voice_filepath in audio_filepaths:
            parent_dir_name = os.path.basename(os.path.dirname(audio_voice_filepath))
            number_str: str = parent_dir_name[14:16]
            performance: str = parent_dir_name[17:21]
            matching_midi_files = [midi_file for midi_file in midi_filepaths if
                                   re.compile(fr".*-{number_str}_{performance}.*").search(midi_file)]
            if len(matching_midi_files) > 1:
                raise RuntimeError(f"Found more than one matching file for audio filename: {audio_voice_filepath}")
            midi_voice_filepath: str = matching_midi_files[0]
            audio_midi_combination.append((audio_voice_filepath, midi_voice_filepath))
        return audio_midi_combination

    @staticmethod
    def create_audio_tsv_1(filepaths_audio_midi: List[Tuple[str, str]], tsv_dir: str) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        audio_filepath: str
        midi_filepath: str
        if not os.path.exists(tsv_dir):
            os.makedirs(tsv_dir)
        for audio_filepath, midi_filepath in filepaths_audio_midi:
            tsv_filepath = os.path.join(tsv_dir, os.path.basename(midi_filepath).replace('.mid', '.tsv'))
            if not os.path.exists(tsv_filepath):
                midi.create_tsv_from_midi(midi_filepath, tsv_filepath)
            result.append((audio_filepath, tsv_filepath))
        return result

    def files(self, group: str) -> List[Tuple]:
        audio_filepaths: List[str] = super().get_filepaths_from_group(self.swd_vocal_wav, group)
        voice_audio_filepaths: List[str] = []
        for path in audio_filepaths:
            if path.__contains__('vocals'):
                voice_audio_filepaths.append(path)
        if len(voice_audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')
        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.swd_csv, '*.csv'))
        assert len(ann_audio_note_filepaths_csv) > 0
        midi_path = midi.save_csv_as_midi(ann_audio_note_filepaths_csv, self.swd_vocal_midi, instrument_arg='voice')
        midi_voice_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))
        files_voice_midi_filepaths: List[Tuple[str, str]] = self.combine_audio_midi(voice_audio_filepaths,
                                                                                    midi_voice_filepaths)
        voice_tsv_filepaths = self.create_audio_tsv_1(files_voice_midi_filepaths, self.swd_vocal_tsv)
        return voice_tsv_filepaths

    def clear_computed(self):
        logging.info(f'Clearing dirs {self.swd_midi} and {self.swd_tsv}.\n'
                     f'This is because clear_computed is set to true. \n'
                     f'They are recomputed. For a faster execution, you have to disable clear_computed.')
        if os.path.exists(self.swd_midi):
            shutil.rmtree(self.swd_midi)
        if os.path.exists(self.swd_tsv):
            shutil.rmtree(self.swd_tsv)
        super().clear_computed()


class WagnerRingDataset(PianoRollAudioDataset):
    wr_midi: str
    wr_csv: str
    wr_tsv: str
    wr_audio_wav: str

    def __init__(self, path='data/WagnerRing_v0-1', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        self.wr_midi = os.path.join(path, '02_Annotations', '_ann_audio_note_midi')
        self.wr_csv = os.path.join(path, '02_Annotations', 'ann_audio_note')
        self.wr_tsv = os.path.join(path, '02_Annotations', '_ann_audio_note_tsv')
        self.wr_audio_wav = os.path.join(path, '01_RawData', 'audio_wav')

        super().__init__(path, groups, sequence_length, seed, device)

    def files(self, group):
        logging.info(f"Loading Files for group {group}, searching in {self.wr_audio_wav}")
        audio_filepaths: List[str] = SchubertWinterreiseDataset.get_filepaths_from_group(self.wr_audio_wav, group)
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.wr_csv, '*.csv'))
        assert len(ann_audio_note_filepaths_csv) > 0

        # save csv as midi
        midi_path = midi.save_nt_csv_as_midi(ann_audio_note_filepaths_csv, self.wr_midi)
        midi_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))

        # combine .wav with .midi
        filepaths_audio_midi: List[Tuple[str, str]] = self._combine_audio_midi(audio_filepaths, midi_filepaths)

        audio_tsv_filepaths = SchubertWinterreiseVoice.create_audio_tsv_1(filepaths_audio_midi, self.wr_tsv)
        return audio_tsv_filepaths

    @classmethod
    def available_groups(cls):
        return ['KeilberthFurtw1952', 'Furtwangler1953', 'Krauss1953', 'Solti1958', 'Karajan1966', 'Bohm1967',
                'Swarowsky1968', 'Boulez1980', 'Janowski1980', 'Levine1987', 'Haitink1988', 'Sawallisch1989',
                'Barenboim1991', 'Neuhold1993', 'Weigle2010', 'Thielemann2011']

    @staticmethod
    def _combine_audio_midi(audio_filepaths: List[str], midi_filepaths: List[str]) -> List[Tuple[str, str]]:
        audio_midi_combination: List[Tuple[str, str]] = []
        for audio_filepath in audio_filepaths:
            basename = os.path.basename(audio_filepath).replace('.wav', '')
            matching_files = [midi_file for midi_file in midi_filepaths if
                              re.compile(fr".*{basename}.*").search(midi_file)]
            if len(matching_files) != 1:
                raise RuntimeError(f"Found different number of matching midi files than expected for: {audio_filepath}")
            midi_filepath: str = matching_files[0]
            audio_midi_combination.append((audio_filepath, midi_filepath))
        return audio_midi_combination


class Bach10Dataset(PianoRollAudioDataset):
    bach10_midi: str
    bach10_csv: str
    bach10_tsv: str
    bach10_audio_wav: str

    def __init__(self, path='data/Bach10', groups=None):
        self.bach10_midi = os.path.join(path, '_ann_audio_note_midi')
        self.bach10_csv = os.path.join(path, 'ann_audio_pitch_CSV')
        self.bach10_tsv = os.path.join(path, '_ann_audio_note_tsv')
        self.bach10_audio_wav = os.path.join(path, 'audio_wav_44100_mono')

        super().__init__(path, groups)

    def __str__(self):
        return 'Bach10'

    @classmethod
    def available_groups(cls):
        return ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    def files(self, group: str):
        logging.info(f"Loading Files for group {group}, searching in {self.bach10_audio_wav}")

        audio_filepaths: List[str] = glob(os.path.join(self.bach10_audio_wav, group + '*' + '*.wav'))
        if len(audio_filepaths) != 1:
            raise RuntimeError(f'Expected one file for group {group}, found {len(audio_filepaths)}.')

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.bach10_csv, group + '*'))
        if len(ann_audio_note_filepaths_csv) != 1:
            raise RuntimeError(
                f'Expected one annotation file for group {group}, found {len(ann_audio_note_filepaths_csv)}.')

        midi_path = midi.save_nt_csv_as_midi(ann_audio_note_filepaths_csv, self.bach10_midi)
        midi_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))

        filepaths_audio_midi: List[Tuple[str, str]] = WagnerRingDataset._combine_audio_midi(audio_filepaths,
                                                                                            midi_filepaths)
        audio_tsv_filepaths = SchubertWinterreiseVoice.create_audio_tsv_1(filepaths_audio_midi, self.bach10_tsv)
        return audio_tsv_filepaths


class PhenicxAnechoicDataset(PianoRollAudioDataset):
    phenicx_anechoic_mixaudio_wav: str
    phenicx_anechoic_annotations: str
    phenicx_anechoic_tsv: str

    def __init__(self, path='data/PHENICX-Anechoic', groups=None):
        self.phenicx_anechoic_mixaudio_wav = os.path.join(path, 'mixaudio_wav_22050_mono')
        self.phenicx_anechoic_annotations = os.path.join(path, 'annotations')
        self.phenicx_anechoic_tsv = os.path.join(path, '_ann_audio_note_tsv')

        super().__init__(path, groups)

    def __str__(self):
        return 'PhenicxAnechoic'

    @classmethod
    def available_groups(cls):
        return ['beethoven', 'bruckner', 'mahler', 'mozart']

    @staticmethod
    def create_audio_tsv(filepaths_audio_midi: List[Tuple[str, str]], tsv_dir: str) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        audio_filepath: str
        midi_filepath: str
        if not os.path.exists(tsv_dir):
            os.makedirs(tsv_dir)
        for audio_filepath, midi_filepath in filepaths_audio_midi:
            tsv_filepath = os.path.join(tsv_dir, os.path.basename(audio_filepath).replace('.wav', '.tsv'))
            if not os.path.exists(tsv_filepath):
                midi.create_tsv_from_midi(midi_filepath, tsv_filepath)
            result.append((audio_filepath, tsv_filepath))
        return result

    def files(self, group):
        logging.info(f'Loading files for group {group}, searching in {self.phenicx_anechoic_mixaudio_wav}')

        audio_filepath: str = os.path.join(self.phenicx_anechoic_mixaudio_wav, group + '.wav')
        midi_filepaths: List[str] = glob(os.path.join(self.phenicx_anechoic_annotations, group, '*.mid'))
        # remove the all.mid file, where all the _o files are included
        midi_filepaths = [f for f in midi_filepaths if not re.compile(fr".*all.mid").search(f)]
        # remove all original files (not warped to the actual recording)
        midi_filepaths = [f for f in midi_filepaths if not re.compile(fr".*_o.mid").search(f)]

        midi_path: str = midi.combine_midi_files(midi_filepaths, os.path.join(self.phenicx_anechoic_annotations, group,
                                                                              'warped_all.mid'))
        audio_tsv_filepaths = self.create_audio_tsv([(audio_filepath, midi_path)], self.phenicx_anechoic_tsv)
        return audio_tsv_filepaths


class RwcDataset(PianoRollAudioDataset):
    rwc_wav: str
    rwc_midi_warped: str
    rwc_tsv: str

    non_piano_ids: List[str] = ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012', '013',
                                '014', '015', '016', '017', '024', '025', '036', '038', '041']
    non_piano_train_ids: List[str] = ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012',
                                      '013', '014', '024', '025']
    non_piano_validation_ids: List[str] = ['041', '038', '036']
    non_piano_test_ids: List[str] = ['015', '016', '017']

    def __init__(self, path='data/RWC', groups=None):
        self.rwc_wav = os.path.join(path, 'wav_22050_mono')
        self.rwc_midi_warped = os.path.join(path, 'MIDI_warped')
        self.rwc_tsv = os.path.join(path, '_ann_audio_note_tsv')

        super().__init__(path, groups)

    def __str__(self):
        return 'Rwc'

    @classmethod
    def available_groups(cls):
        return ['rwc', 'non-piano', 'non-piano-train', 'non-piano-validation', 'non-piano-test']

    def files(self, group):
        logging.info(f'Loading files for group {group}, searching in {self.rwc_wav}')
        audio_filepaths: List[str] = glob(os.path.join(self.rwc_wav, '*.wav'), recursive=False)
        midi_filepaths: List[str] = glob(os.path.join(self.rwc_midi_warped, '*.mid'), recursive=False)

        if len(audio_filepaths) == 0 or len(midi_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        if 'non-piano-train' in group:
            audio_filepaths = [f for f in audio_filepaths if any(id_ in f for id_ in self.non_piano_train_ids)]
            midi_filepaths = [f for f in midi_filepaths if any(id_ in f for id_ in self.non_piano_train_ids)]
        elif 'non-piano-validation' in group:
            audio_filepaths = [f for f in audio_filepaths if any(id_ in f for id_ in self.non_piano_validation_ids)]
            midi_filepaths = [f for f in midi_filepaths if any(id_ in f for id_ in self.non_piano_validation_ids)]
        elif 'non-piano-test' in group:
            audio_filepaths = [f for f in audio_filepaths if any(id_ in f for id_ in self.non_piano_test_ids)]
            midi_filepaths = [f for f in midi_filepaths if any(id_ in f for id_ in self.non_piano_test_ids)]

        # combine .wav with .mid
        filepaths_audio_midi: List[Tuple[str, str]] = WagnerRingDataset._combine_audio_midi(audio_filepaths,
                                                                                            midi_filepaths)
        audio_tsv_filepaths = SchubertWinterreiseVoice.create_audio_tsv_1(filepaths_audio_midi, self.rwc_tsv)
        return audio_tsv_filepaths


class TriosDataset(PianoRollAudioDataset):
    # Trios is exclusively used for testing in the 'comparing' paper -> not using here for training!
    @classmethod
    def available_groups(cls):
        pass

    def files(self, group):
        pass


class ChoralSingingDataset(PianoRollAudioDataset):
    csd_audio_dir: str
    csd_midi_mixed: str
    csd_tsv: str

    def __init__(self, path='data/ChoralSingingDataset', groups=None):
        self.csd_audio_dir = os.path.join(path, 'mixaudio_wav_22050_mono')
        self.csd_midi_mixed = os.path.join(path, '_ann_audio_note_midi')
        self.csd_tsv = os.path.join(path, '_ann_audio_note_tsv')
        super().__init__(path, groups)

    def __str__(self):
        return 'ChoralSingingDataset'

    @classmethod
    def available_groups(cls):
        return ['Bruckner_LocusIste', 'Guerrero_NinoDios', 'Traditional_ElRossinyol']

    def files(self, group):
        logging.info(f'Loading files for group {group}, searching in {self.path}')
        audio_filepaths: List[str] = glob(os.path.join(self.csd_audio_dir, '*' + group + '*.wav'))
        if len(audio_filepaths) != 5:
            raise RuntimeError(f'Expected exactly 5 files for group {group}, found {len(audio_filepaths)} files.')

        midi_filepaths: List[str] = glob(
            os.path.join(self.path, 'ChoralSingingDataset', 'CSD_' + group, 'midi', '*.mid'), recursive=False)
        if len(midi_filepaths) != 4:
            raise RuntimeError(f'Expected four midi files for group {group}, found {len(midi_filepaths)} files.')

        midi_sorted: Dict = {}
        for midifile in midi_filepaths:
            if 'alt' in midifile:
                midi_sorted['alt'] = midifile
            elif 'sop' in midifile:
                midi_sorted['sop'] = midifile
            elif 'ten' in midifile:
                midi_sorted['ten'] = midifile
            elif 'bas' in midifile:
                midi_sorted['bas'] = midifile
            else:
                raise RuntimeError()

        filepaths_audio_midi: List[Tuple[str, str]] = []

        for audio_file in audio_filepaths:
            if 'alt' in audio_file:
                noalt_midi = midi.combine_midi_files([midi_sorted['sop'], midi_sorted['ten'], midi_sorted['bas']],
                                                     os.path.join(self.csd_midi_mixed, group + 'noalt.mid'))
                filepaths_audio_midi.append((audio_file, noalt_midi))
            elif 'sop' in audio_file:
                nosop_midi = midi.combine_midi_files([midi_sorted['alt'], midi_sorted['ten'], midi_sorted['bas']],
                                                     os.path.join(self.csd_midi_mixed, group + 'nosop.mid'))
                filepaths_audio_midi.append((audio_file, nosop_midi))
            elif 'ten' in audio_file:
                noten_midi = midi.combine_midi_files([midi_sorted['sop'], midi_sorted['alt'], midi_sorted['bas']],
                                                     os.path.join(self.csd_midi_mixed, group + 'noten.mid'))
                filepaths_audio_midi.append((audio_file, noten_midi))
            elif 'bas' in audio_file:
                nobas_midi = midi.combine_midi_files([midi_sorted['sop'], midi_sorted['alt'], midi_sorted['ten']],
                                                     os.path.join(self.csd_midi_mixed, group + 'nobas.mid'))
                filepaths_audio_midi.append((audio_file, nobas_midi))
            else:
                all_midi = midi.combine_midi_files(
                    [midi_sorted['sop'], midi_sorted['alt'], midi_sorted['ten'], midi_sorted['bas']],
                    os.path.join(self.csd_midi_mixed, group + 'all.mid'))
                filepaths_audio_midi.append((audio_file, all_midi))

        audio_tsv_filepaths = SchubertWinterreiseVoice.create_audio_tsv_1(filepaths_audio_midi, self.csd_tsv)
        return audio_tsv_filepaths


class MusicNetDataset(PianoRollAudioDataset):
    mun_audio: str
    mun_generated_midi_annotations: str
    mun_tsv: str

    MUN_ANNOTATION_SAMPLERATE: int = 44100

    validation_set_files = ['1729', '1733', '1755', '1756', '1765', '1766', '1805', '1807', '1811', '1828', '1829',
                            '1932', '1933', '2081', '2082', '2083', '2157', '2158', '2167', '2186', '2194', '2221',
                            '2222', '2289', '2315', '2318', '2341', '2342', '2480', '2481', '2629', '2632', '2633']
    """
    MuN validation files, copied from 
    """

    test_set_files: Dict = {
        'MuN-3-test': ['2303', '1819', '2382'],
        'MuN-10-test': ['2303', '1819', '2382', '2298', '2191', '2556', '2416', '2628', '1759', '2106'],
        'MuN-10-var-test': ['2303', '1819', '2382', '2298', '2191', '2556', '2416', '2629', '1759', '2106'],
        'MuN-10-slow-test': ['2302', '1818', '2383', '2293', '2186', '2557', '2415', '2627', '1758', '2105'],
        'MuN-10-fast-test': ['2310', '1817', '2381', '2296', '2186', '2555', '2417', '2626', '1757', '2104'],
        'MuN-36-cyc-test': ['2302', '2303', '2304', '2305',
                            '1817', '1818', '1819',
                            '2381', '2382', '2383', '2384',
                            '2293', '2294', '2295', '2296', '2297', '2298',
                            '2186', '2191',
                            '2555', '2556', '2557',
                            '2415', '2416', '2417',
                            '2626', '2627', '2628', '2629',
                            '1757', '1758', '1759', '1760',
                            '2104', '2105', '2106']
    }

    non_piano_files = ['2219', '2288', '2294', '2241', '2186', '2296', '2204', '2203', '2191', '2289', '2217', '2298',
                       '2220', '2295', '2297', '2659', '2244', '2222', '2242', '2218', '2202', '2293', '2221', '2243',
                       '2156', '2131', '2117', '2140', '2154', '2147', '2155', '2119', '2127', '2116', '2138', '2118',
                       '2157', '2105', '2106', '2104', '2179', '2177', '2180', '2178', '2376', '2483', '2415', '2433',
                       '2560', '2504', '2497', '2507', '2506', '2562', '2377', '2494', '2481', '2314', '2432', '2403',
                       '2383', '2313', '2315', '2417', '2365', '2480', '2431', '2368', '2482', '2381', '2382', '2379',
                       '2505', '2622', '2621', '2384', '2366', '2451', '2416', '1918', '1922', '1919', '1933', '1931',
                       '1932', '1916', '1923', '1742', '2081', '2079', '2078', '2075', '2080', '2083', '2077', '2082',
                       '2076', '1790', '1788', '1812', '1805', '1807', '1859', '1789', '1824', '1793', '1819', '1811',
                       '1835', '1792', '1822', '1791', '1818', '1813', '1817']
    non_piano_validation_files = list(set(non_piano_files) & set(validation_set_files))
    non_piano_train_files = list(set(non_piano_files) - set(non_piano_validation_files))

    def __init__(self, path='data/MusicNet', groups=None):
        self.mun_audio = os.path.join(path, 'musicnet')
        self.mun_generated_midi_annotations = os.path.join(path, '_musicnet_generated_midi')
        self.mun_tsv = os.path.join(path, '_ann_audio_note_tsv')

        super().__init__(path, groups)

    def __str__(self):
        return 'MusicNet'

    def save_mun_csv_as_midi(self, csv_file, midi_path) -> str:
        if not os.path.exists(midi_path):
            os.mkdir(midi_path)

        csv_annotations: pd.DataFrame = pd.read_csv(csv_file, sep=',')
        midi_filename = os.path.basename(csv_file.replace('.csv', '.mid'))
        midi_filepath = os.path.join(midi_path, midi_filename)
        if os.path.exists(midi_filepath):
            return str(midi_filepath)

        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        for idx, row in csv_annotations.iterrows():
            onset: float = row[0] / self.MUN_ANNOTATION_SAMPLERATE
            offset: float = row[1] / self.MUN_ANNOTATION_SAMPLERATE
            pitch: int = int(row[3])
            note = pretty_midi.Note(start=onset, end=offset, pitch=pitch, velocity=64)
            piano.notes.append(note)
        file: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
        file.instruments.append(piano)
        file.write(midi_filepath)

        return str(midi_filepath)

    @classmethod
    def available_groups(cls):
        return ['MuN-3-train', 'MuN-3-test', 'MuN-10-train', 'MuN-10-test', 'MuN-10-var-train', 'MuN-10-var-test',
                'MuN-10-slow-train', 'MuN-10-slow-test', 'MuN-10-fast-train', 'MuN-10-fast-test',
                'MuN-36-cyc-train', 'MuN-36-cyc-test', 'MuN-validation',
                'MuN-non-piano-tr', 'MuN-non-piano-val']

    def files(self, group):
        logging.info(f'Loading files for group {group}, searching in {self.mun_audio}')
        all_audio_filepaths = glob(os.path.join(self.mun_audio, '**', '*.wav'), recursive=True)
        audio_filepaths_filtered: List[str] = []

        # Filter audio files based on MuN groups defined above
        if 'test' in group:
            test_labels: List[str] = self.test_set_files[group]
            for filepath in all_audio_filepaths:
                if any(test_label in filepath for test_label in test_labels):
                    audio_filepaths_filtered.append(filepath)
        elif 'train' in group:
            group_test = group[:-5] + 'test'
            test_labels: List[str] = self.test_set_files[group_test] + self.validation_set_files
            for filepath in all_audio_filepaths:
                if not any(test_label in filepath for test_label in test_labels):
                    audio_filepaths_filtered.append(filepath)
        elif 'validation' in group:
            for filepath in all_audio_filepaths:
                if any(validation_label in filepath for validation_label in self.validation_set_files):
                    audio_filepaths_filtered.append(filepath)
        elif 'non-piano-tr' in group:
            for filepath in all_audio_filepaths:
                if any(train_label in filepath for train_label in self.non_piano_train_files):
                    audio_filepaths_filtered.append(filepath)
        elif 'non-piano-val' in group:
            for filepath in all_audio_filepaths:
                if any(val_label in filepath for val_label in self.non_piano_validation_files):
                    audio_filepaths_filtered.append(filepath)
        else:
            raise ValueError(f'Specified unknown group for this dataset. Specified: {group}')

        if len(audio_filepaths_filtered) < 2:
            raise RuntimeError(
                f'Received unexpected number of files for group {group}, found {len(audio_filepaths_filtered)}')

        filepaths_audio_midi: List[Tuple[str, str]] = []
        for file in tqdm(audio_filepaths_filtered, desc='Converting MuN csv to midi and tsv.'):
            identifier = os.path.basename(file)[:-4]
            csv_files = glob(os.path.join(self.mun_audio, '**', identifier + '*.csv'), recursive=True)
            if len(csv_files) != 1:
                raise RuntimeError(f'Expected 1 file for {file}, got {len(csv_files)}')
            midi_filepath = self.save_mun_csv_as_midi(csv_files[0], self.mun_generated_midi_annotations)
            filepaths_audio_midi.append((file, midi_filepath))
        audio_tsv_filepaths = SchubertWinterreiseVoice.create_audio_tsv_1(filepaths_audio_midi, self.mun_tsv)
        return audio_tsv_filepaths

    def clear_computed(self):
        logging.info(f'Clearing directories: {self.mun_tsv}, {self.mun_generated_midi_annotations}')
        if os.path.isdir(self.mun_tsv) and os.path.isdir(self.mun_generated_midi_annotations):
            shutil.rmtree(self.mun_tsv)
            shutil.rmtree(self.mun_generated_midi_annotations)
        else:
            raise RuntimeError(f'Could not clear directories {self.mun_tsv}, {self.mun_generated_midi_annotations}')
        super().clear_computed()
