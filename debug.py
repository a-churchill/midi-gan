from typing import List, Union

import matplotlib.pyplot as plt
import music21
import torch


def plot_losses(losses, legend=None):
    """Plots the losses over each epoch"""
    if legend is None:
        legend = [
            "real_loss",
            "fake_loss",
            "gan_loss",
            "feature_map_loss",
            "result_loss",
        ]
    plt.figure(figsize=(8, 8))
    plt.title("Losses")
    plt.plot(losses)
    plt.legend(legend)
    plt.show()


def plot_midi(tensor: torch.Tensor):
    """Plots a generated MIDI tensor"""
    fig = plt.figure()
    plt.title("Score")
    data = tensor.cpu().detach().numpy()
    plt.imshow(data, origin="lower")
    return fig


def tensor_to_midi(tensor: torch.Tensor) -> music21.stream.Stream:
    """Converts a tensor (representing a score) to a `music21` stream.

    Args:
        tensor (torch.Tensor): HxW tensor, where H is the number of MIDI notes (128) and W is the
        number of 16th notes in the score

    Returns:
        music21.stream.Stream: stream representing the given score
    """
    nonzero_tensor = torch.nonzero(tensor)
    nonzero = {
        (nonzero_tensor[i, 0].item(), nonzero_tensor[i, 1].item())
        for i in range(nonzero_tensor.size(0))
    }

    result = music21.stream.Stream()
    result.append(music21.instrument.Piano())

    # template duration, used for rests and notes
    duration = music21.duration.Duration()
    duration.quarterLength = 1 / 4

    # one element per sixteenth note timestep
    chord_by_timestep: List[Union[music21.note.Rest, music21.chord.Chord]] = [
        music21.note.Rest(duration=duration) for _ in range(tensor.size(1))
    ]

    for pitch, timestep in nonzero:
        if pitch == 0:
            continue

        current_chord = (
            music21.chord.Chord(duration=duration)
            if chord_by_timestep[timestep].isRest
            else chord_by_timestep[timestep]
        )
        note = music21.note.Note()
        note.pitch = music21.pitch.Pitch(midi=pitch)
        current_chord.add(note)
        chord_by_timestep[timestep] = current_chord

    result.append(chord_by_timestep)
    result.extendTies()  # this joins identical notes

    return result


# def tensor_to_midi(tensor: torch.Tensor) -> music21.stream.Stream:
#     """Converts a tensor (representing a 4-part score) to a `music21` stream.

#     Args:
#         tensor (torch.Tensor): 4xN tensor, where N is the number of 16th notes in the score

#     Returns:
#         music21.stream.Stream: stream representing the given score
#     """

#     def part_to_midi(part: torch.Tensor, name: str) -> music21.stream.Part:
#         """Converts a single list of pitches into a `music21` `Part`.

#         Args:
#             part (torch.Tensor): 1-dimensional tensor, with pitches encoded as `process_score` does
#             name (str): the name of the part

#         Returns:
#             music21.stream.Part: a `music21` `Part`
#         """
#         result = music21.stream.Part(id=name)
#         instrument = music21.instrument.Piano()
#         instrument.instrumentName = f"Piano: {name}"
#         instrument.midiProgram = 0
#         result.append(instrument)
#         pitch_length = 1
#         for i in range(part.size(0)):
#             pitch = part[i].item()
#             next_pitch = part[i + 1].item() if i + 1 < part.size(0) else None
#             if pitch == next_pitch:
#                 pitch_length += 1
#             else:
#                 # this pitch is the last one of its type
#                 if pitch == -1:
#                     rest = music21.note.Rest()
#                     rest.duration.quarterLength = pitch_length / 4
#                     result.append(rest)
#                 else:
#                     note = music21.note.Note()
#                     note.duration.quarterLength = pitch_length / 4
#                     note.pitch = music21.pitch.Pitch(midi=pitch)
#                     result.append(note)

#                 pitch_length = 1
#         return result

#     return music21.stream.Stream(
#         [
#             part_to_midi(tensor[0, :], "Soprano"),
#             part_to_midi(tensor[1, :], "Alto"),
#             part_to_midi(tensor[2, :], "Tenor"),
#             part_to_midi(tensor[3, :], "Bass"),
#         ]
#     )
