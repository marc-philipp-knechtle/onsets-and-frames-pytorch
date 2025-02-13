import json
import logging
import os
import re
import shutil
from abc import abstractmethod
from glob import glob
from typing import List, Tuple, Dict

import librosa
import numpy as np
import soundfile
from torch import Tensor

from torch.utils.data import IterableDataset
from tqdm import tqdm

from . import midi
from .constants import *
from .midi import parse_midi

dataset_definitions = {
    'maestro_training': lambda: MAESTRO(groups=['train'], sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'maestro_validation': lambda: MAESTRO(groups=['validation'], sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'winterreise_training': lambda: SchubertWinterreiseDataset(groups=['FI55', 'FI66', 'FI80', 'OL06', 'QU98', 'TR99'],
                                                               sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'winterreise_validation': lambda: SchubertWinterreiseDataset(groups=['AL98'],
                                                                 sequence_length=DEFAULT_SEQUENCE_LENGTH),
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
    'b10_train': lambda: Bach10Dataset(groups=['01', '02', '03', '04'], sequence_length=DEFAULT_SEQUENCE_LENGTH),
    'b10_validation': lambda: Bach10Dataset(groups=['05', '06'], sequence_length=DEFAULT_SEQUENCE_LENGTH)
}


class PianoRollAudioDataset(IterableDataset):
    path: str
    groups: List[str]
    sequence_length: int
    device: str
    random: np.random.RandomState
    data: List[Dict]

    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
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
                     "The .pt files are created again for this run. For a faster execution you have to disable"
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
            return torch.load(saved_data_path)

        audio: np.ndarray
        audio, sr = soundfile.read(audio_path, dtype='int16')
        # Conversion to fload see:
        # https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
        audio: np.ndarray = np.array(audio).astype(np.float32)
        if len(audio.shape) > 1:
            # Convert Stereo to Mono
            logging.warning('Audio is twodimensional - this is a stereo file! Converting to mono!')
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
            metadata = json.load(open(os.path.join(self.path, 'maestro-v1.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if
                            row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi_filename) for
                     audio, midi_filename in files]

        result = []
        audio_path: str
        midi_path: str
        for audio_path, midi_path in files:
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

    def __init__(self,
                 path='data/Schubert_Winterreise_Dataset_v2-1', groups=None, sequence_length=None, seed=42,
                 device=DEFAULT_DEVICE):
        # adding underscore to symbolize that these annotations are computationally created
        self.swd_midi = os.path.join(path, '02_Annotations', '_ann_audio_note_midi')
        self.swd_csv = os.path.join(path, '02_Annotations', 'ann_audio_note')
        self.swd_tsv = os.path.join(path, '02_Annotations', '_ann_audio_note_tsv')
        self.swd_audio_wav = os.path.join(path, '01_RawData', 'audio_wav')

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
        audio_filepaths: List[str] = self.get_filepaths_from_group(self.swd_audio_wav, group)
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

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

    def __init__(self, path='data/Bach10', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        self.bach10_midi = os.path.join(path, '_ann_audio_note_midi')
        self.bach10_csv = os.path.join(path, 'ann_audio_pitch_CSV')
        self.bach10_tsv = os.path.join(path, '_ann_audio_note_tsv')
        self.bach10_audio_wav = os.path.join(path, 'audio_wav_44100_mono')

        super().__init__(path, groups, sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    def files(self, group: str):
        logging.info(f"Loading Files for group {group}, searching in {self.bach10_audio_wav}")

        audio_filepaths: List[str] = glob(os.path.join(self.bach10_audio_wav, group + '*'))
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
