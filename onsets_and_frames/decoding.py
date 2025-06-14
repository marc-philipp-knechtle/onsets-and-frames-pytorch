import numpy as np
import torch

def extract_notes_from_frames(frames, threshold):
    """
    With this, you can use just the frame output to extract the notes
    This does not use the onsets.
    We just use a custom threshold to specify after what number of notes we consider a note to be detected.
    """
    frames = (frames > threshold).cpu().to(torch.uint8) # sets each value to 1 if it is above the threshold, 0 otherwis
    pitches = []
    intervals = []
    velocities = []

    value1 = frames[:1, :]
    """
    shape(88, 1)
    """
    value2 = frames[1: , :]
    """
    shape(n_frames - 1, 88) 
    -> remove the first row of the frames tensor
    """
    value3 = frames[:-1, :]
    """
    -> removes the last row of the tensor
    """
    value4 = value2 - value3
    """
    difference in each row
    current row minus next row
    0 if current detected as 1  and next detected as one
    1 if current is 1 and next is 0
    -1 if currentis 0 and the next is 1
    """

    # We cannot use the same approach as in def extract_notes() here -> the frame output is completely different
    value1_numpy = value1.numpy()
    value2_numpy = value2.numpy()
    value3_numpy = value3.numpy()
    value4_numpy = value4.numpy()
    frames_numpy = frames.numpy()

    onsets_from_frames = torch.cat([frames[:1, :], frames[1:, :] - frames[:-1, :]], dim=0) == 1
    onsets_from_frames_np = onsets_from_frames.numpy()
    # we need to create this to only query the first element of each frame start

    for nonzero in onsets_from_frames.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        while frames[offset, pitch].item():
            offset += 1
            if offset == frames.shape[0]: # = we reach the end of the prediction
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(70)
    return np.array(pitches), np.array(intervals), np.array(velocities)




def extract_notes(onsets, frames, velocity=None, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches:    np.ndarray of bin_indices
                shape: (<length>, 1)
                To my understanding, these are the pitch values for each time index
    intervals:  np.ndarray of rows containing (onset_index, offset_index)
                shape: (<length>, 2)
                Start and end of each note
    velocities: np.ndarray of velocity vales
                shape: (<length>, 1)
                Velocity value for each time index
    """
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    # torch.cat = concatenates tensors. Requirement: each tensor has the same shape!
    # onsets[:1, :] = first row, keeping all columns (=time bin 0 with all possible key values)
    # onsets[1:, :] - onsets[:-1, :] = subtracts each row of onsets from the next row, creating the difference
    # This is true if the current index detects an onset and the next index does not.
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1
    """
    Tensor of False where there is not an onset, true wherer there is
    shape: [frames, bins] (bins=88, because of piano keys)
    """

    if velocity is None:
        velocity = torch.full_like(frames, fill_value=60)

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero(): # .nonzero() returns a tuple containing the indices of nonzero items
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item(): # as long as there is an onset which is still detected
            if onsets[offset, pitch].item(): # if there is still an onset detected
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]: # if we reach the end of the detection
                break

        if offset > onset: # If we have detected sth
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        if pitch >= 88:
            continue
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs
