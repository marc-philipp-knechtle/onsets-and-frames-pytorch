import logging
import math
import multiprocessing
import os.path
import sys
from typing import Tuple, List

import mido
import pandas as pd
import pretty_midi
import numpy as np
import collections

import scipy.io.wavfile
from joblib import Parallel, delayed
from mir_eval.util import hz_to_midi
from tqdm import tqdm


def parse_midi(path: str, global_key_offset: int = 0) -> np.ndarray:
    """
    open midi file and return np.array() of (onset, offset, note, velocity) rows
    Args:
        path: path to midi file
        global_key_offset: sometimes
    Returns: np.ndarray() with (onset, offset, note, velocity) for each note
    """

    midi = mido.MidiFile(path)

    if len(midi.tracks) > 2:
        logging.warning('Multiple midi tracks detected. Found Tracks:')
        for track in midi.tracks:
            logging.warning(track.name)
        logging.warning('The Processing of these files handles all tracks as part of the piano part.\n'
                        'There is no differentiation.')

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        # This was used to skip the vocal annotations for a pure piano training
        # channel: int = getattr(message, 'channel', 0)
        # if channel == 1:
        #     continue

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note + global_key_offset,
                         velocity=velocity,
                         sustain=sustain)
            events.append(event)

    notes: List[Tuple] = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:]
                          if n['type'] == 'sustain_off' or n['note'] == onset['note'] or n is events[-1])

        note: Tuple = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)


def save_midi(path: str, pitches: np.ndarray, intervals: np.ndarray, velocities) -> pretty_midi.PrettyMIDI:
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_time, offset_time)
    velocities: list of velocity values
    """

    midifile = create_midi(intervals, pitches, velocities)
    audio_data: np.ndarray = midifile.synthesize()
    # todo replace this with an implementation of pyfluidsynth
    #  (which can use other sounds compared to the default sine wave)
    scipy.io.wavfile.write(os.path.join(os.path.dirname(path), os.path.basename(path) + '.wav'), 44100,
                           audio_data)
    midifile.write(path)
    return midifile


def create_midi(intervals: np.ndarray, pitches: np.ndarray, velocities: np.ndarray):
    # Remove overlapping intervals (end time should be smaller of equal start time of next note on the same pitch)
    intervals_dict = collections.defaultdict(list)
    for i in range(len(pitches)):
        pitch = int(round(hz_to_midi(pitches[i])))
        intervals_dict[pitch].append((intervals[i], i))
    _check_pitch_time_intervals(intervals_dict)
    piano: pretty_midi.Instrument = _create_piano_midi(intervals_dict, pitches, velocities)
    file: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
    file.instruments.append(piano)
    return file


def save_np_arr_as_midi(midi_arr: np.ndarray, path: str):
    """
    Converts Array of form [(onset_time, offset_time, note, velocity), ...] to MIDI file and saves it.
    This method is mainly used for debugging purposes
    Args:
        midi_arr: Array of form [(onset_time, offset_time, note, velocity), ...]
        path: path to save the MIDI file
    Returns: Nothing lol
    """
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    onset: Tuple
    for onset in midi_arr:
        if math.isnan(onset[2]):
            continue
        velocity = int(onset[3])
        pitch = int(onset[2])
        end_time = onset[1]
        start_time = onset[0]
        note = pretty_midi.Note(start=start_time, end=end_time, velocity=velocity, pitch=pitch)
        piano.notes.append(note)
    file: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
    file.instruments.append(piano)
    file.write(path)


def save_csv_as_midi(csv_filenames: List[str], path: str, instrument_arg: str = 'all') -> str:
    """
    Args:
        csv_filenames: csv files in the format: [onset_time, offset_time, pitch, pitchclass, instrument]
        path: where the MIDI files are stored
        instrument_arg: which instruments to save in the resulting midi
    Returns: path where the MIDI files are saved
    """
    if not os.path.exists(path):
        os.mkdir(path)
    csv_filename: str
    for csv_filename in csv_filenames:
        # process single file
        ann_audio_note: pd.DataFrame = pd.read_csv(csv_filename, sep=';')
        ann_audio_filepath = os.path.join(path, os.path.basename(csv_filename.replace('.csv', '.mid')))

        if os.path.exists(ann_audio_filepath):
            continue

        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        # Another option would be to use Synth Choir, however Oohs sound more appropriate as in Schubert, there is also
        # only a single voice
        voice_program = pretty_midi.instrument_name_to_program('Voice Oohs')
        voice = pretty_midi.Instrument(program=voice_program)

        for idx, row in ann_audio_note.iterrows():
            onset: float = row['start']
            offset: float = row['end']
            pitch: int = row['pitch']
            instrument: str = row['instrument']
            note = pretty_midi.Note(start=onset, end=offset, pitch=pitch, velocity=64)
            if instrument == 'piano':
                piano.notes.append(note)
            elif instrument == 'voice':
                voice.notes.append(note)
            else:
                raise ValueError(f'Unknown instrument {instrument}')
        file: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
        if instrument_arg == 'all':
            file.instruments.append(piano)
            file.instruments.append(voice)
        elif instrument_arg == 'voice':
            file.instruments.append(voice)
        elif instrument_arg == 'piano':
            file.instruments.append(piano)
        else:
            raise ValueError(f'Unknown instrument {instrument_arg}')
        file.write(ann_audio_filepath)
    return path


def create_tsv_from_midi(midi_filepath: str, tsv_filepath: str):
    midi_filename: str = os.path.basename(midi_filepath)
    logging.debug(f'Parsing midi file: {os.path.basename(midi_filename)}.')
    midifile: np.ndarray = parse_midi(midi_filepath)

    # With this statement, you can verify that all the midi information has been converted to tsv correctly
    # save_np_arr_as_midi(midifile, '/tmp/tmp.mid')

    # For some reason pycharm expects an int value in np.savetxt() midi is ofc not an int value.
    # But this error is from pycharm. Therefore, the inspection is disabled here.
    # noinspection PyTypeChecker
    np.savetxt(fname=tsv_filepath, X=midifile, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')

    # Debugging: In case that there is some doubt that the tsv was correctly created -> create again midi from tsv
    # save_np_arr_as_midi(midifile, str(os.path.join(os.path.dirname(tsv_filepath), midi_filename + '.mid')))


def _check_pitch_time_intervals(intervals_dict):
    pitch: int
    for pitch in intervals_dict:
        """
        Describes the list for a single pitch
        : List[Tuple(starttime, endtime), bin time: int]
        I think the bin time refers only to the start, but I'm not sure on this
        """
        interval_list: List[Tuple[List[2], int]] = intervals_dict[pitch]
        interval_list.sort(key=lambda x: x[0][0])
        for i in range(len(interval_list) - 1):
            end_current_interval_seconds = interval_list[i][0][1]
            start_next_interval_seconds = interval_list[i + 1][0][0]
            if end_current_interval_seconds >= start_next_interval_seconds:
                logging.warning(
                    f'End time should be smaller of equal start time of next note on the same pitch.\n'
                    f'Current Pitch End: {end_current_interval_seconds}\n'
                    f'Next pitch Start: {start_next_interval_seconds}\n'
                    f'Correcting with end of current = '
                    f'{min(end_current_interval_seconds, start_next_interval_seconds)}')
                interval_list[i][0][1] = min(end_current_interval_seconds, start_next_interval_seconds)


def _create_piano_midi(intervals_dict, pitches, velocities) -> pretty_midi.Instrument:
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    for pitch in intervals_dict:
        interval_list = intervals_dict[pitch]
        for interval, i in interval_list:
            pitch = int(round(hz_to_midi(pitches[i])))
            # I don't know why we have velocities[i] < 0, but this led to an error
            # The sample did sound correctly. I assume that this was because of the annotations generated by sleeter
            # Sample with this error Schubert_D911-01_HU33
            # Todo check and maybe annotate sleeter for this
            velocity = int(127 * min(velocities[i], 1)) if velocities[i] >= 0 else 0
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=interval[0], end=interval[1])
            piano.notes.append(note)
    return piano


if __name__ == '__main__':
    """
    Creates tsv files for midi file
    """


    def process(input_file, output_file):
        midi_data = parse_midi(input_file)
        # see explanation in dataset.py for np.savetxt
        # noinspection PyTypeChecker
        np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')


    def files():
        for input_file in tqdm(sys.argv[1:]):
            if input_file.endswith('.mid'):
                output_file = input_file[:-4] + '.tsv'
            elif input_file.endswith('.midi'):
                output_file = input_file[:-5] + '.tsv'
            else:
                print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
                continue

            yield input_file, output_file


    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process)(in_file, out_file) for in_file, out_file in files())
