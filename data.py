from pathlib import Path
from typing import List, Union

import music21
import torch
from torch.utils.data import DataLoader, Dataset

from parameters import BATCH_SIZE, CREATIVE_DIMENSION, MIDI_NOTES

# mapping from pitch to index to set to 1 in 1-hot vector encoding of pitch
pitch_to_index = {
    "C": 0,
    "C#": 1,
    "D-": 1,
    "D": 2,
    "D#": 3,
    "E-": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G-": 6,
    "G": 7,
    "G#": 8,
    "A-": 8,
    "A": 9,
    "A#": 10,
    "B-": 10,
    "B": 11,
}


def is_almost_int(num: float) -> bool:
    """Checks if `num` is almost an int (within some small delta of an integer)

    Args:
        num (float): a number to check

    Returns:
        bool: `True` if `num` is almost an int, `False` otherwise
    """
    return num - int(num) < 0.0001


def process_part(part: Union[music21.stream.Part, None]) -> List[int]:
    """Turns a `Part` (from `music21`) into a list of MIDI pitches. Note that 0 is a special pitch,
    meant to represent a rest.

    Args:
        part (Union[music21.stream.Part, None]): part to convert

    Returns:
        List[int]: pitches from the given part
    """
    if part is None:
        return None
    notes = part.recurse().notesAndRests

    pitches = []
    for note in notes:
        sixteenth_length = float(note.quarterLength) * 4
        if not is_almost_int(sixteenth_length):
            # ignore parts that don't use sixteenth note granularity
            return None

        sixteenth_length = int(sixteenth_length)

        pitch_number = 0 if note.isRest else note.pitch.midi
        pitches.extend([pitch_number] * sixteenth_length)

    assert len(pitches) == int(part.duration.quarterLength * 4)
    return pitches


def process_score(path: Path) -> torch.Tensor:
    """Generates a tensor representing the score from a path to a file parseable by `music21`

    Args:
        path (Path): path to file to load the score

    Returns:
        torch.Tensor: A tensor with dimensions HxW, where H is the number of possible MIDI notes
        (128) and W is the number of sixteenth notes in the score.
    """
    score = music21.converter.parse(path)
    parts = {part.id: part for part in score.getElementsByClass(music21.stream.Part)}

    soprano = process_part(parts.get("Soprano", None))
    alto = process_part(parts.get("Alto", None))
    tenor = process_part(parts.get("Tenor", None))
    bass = process_part(parts.get("Bass", None))

    if soprano is None or alto is None or tenor is None or bass is None:
        return None

    assert len(soprano) == len(alto)
    assert len(soprano) == len(tenor)
    assert len(soprano) == len(bass)

    result = torch.zeros((MIDI_NOTES, len(soprano)))
    for i, notes in enumerate(zip(soprano, alto, tenor, bass)):
        result[notes, i] = 1.0

    return result


class MIDIDataset(Dataset):
    """Dataset of a given artist's works (from the `music21` corpus)"""

    def __init__(self, artist: str):
        paths = list(
            filter(
                lambda path: str(path).endswith(".mxl"),
                music21.corpus.getComposer(artist),
            )
        )
        scores_with_none = [process_score(path) for path in paths]
        self.all_measures = torch.concat(
            tuple(
                score_tensor[
                    :, 0 : (score_tensor.size(1) // 16) * 16 - 16
                ]  # chop off last measure
                for score_tensor in scores_with_none
                if score_tensor is not None
            ),
            dim=1,
        )

    def __len__(self) -> int:
        return self.all_measures.size(1) // (
            16 * 2
        )  # we return measures in (last_measure, this_measure) pairs

    def __getitem__(self, i: int) -> torch.Tensor:
        """Returns (last_measure, this_measure, chord_tensor) tuple"""
        last_measure = self.all_measures[:, i * 32 : i * 32 + 16]
        current_measure = self.all_measures[:, i * 32 + 16 : (i + 1) * 32]
        first_pitch = music21.pitch.Pitch(
            midi=current_measure[:, 0].nonzero().min().item()
        )
        chord_tensor = torch.zeros((CREATIVE_DIMENSION,))
        chord_tensor[pitch_to_index[first_pitch.name]] = 1

        return last_measure, current_measure, chord_tensor


def get_dataloader(dataset: MIDIDataset) -> DataLoader:
    """Gets a dataloader for the given data set"""
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


dataloader = get_dataloader(MIDIDataset("bach"))
