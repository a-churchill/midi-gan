from typing import List
import torch

from data import pitch_to_index
from gan import GAN
from parameters import CREATIVE_DIMENSION, DEVICE, MIDI_NOTES, NOISE_DIMENSION


def model_result_to_tensor(model_result: torch.Tensor) -> torch.Tensor:
    """Converts a model result (with continuous results in any range) to a discrete result with
    only 4 nonzero values
    """
    top_values, _ = torch.kthvalue(model_result, MIDI_NOTES - 3, dim=0, keepdim=True)
    return (model_result > top_values).to(torch.float)


def run_model(
    model: GAN, initial_measure: torch.Tensor, chords_input: List[str]
) -> torch.Tensor:
    """Runs the model given an initial measure and a set of chords.

    Args:
        model (GAN): trained model
        initial_measure (torch.Tensor): seed measure
        chords_input (List[str]): a list of pitch names (e.g. "A", "B-", "C#") - there will be one
        measure generated for every pitch name

    Returns:
        torch.Tensor: a generated piece
    """
    initial_measure = initial_measure.to(DEVICE)
    model = model.to(DEVICE)
    model.eval()

    generated_length = len(chords_input)
    chords_tensor = torch.zeros(generated_length, CREATIVE_DIMENSION, device=DEVICE)
    for i, chord in enumerate(chords_input):
        chords_tensor[i, pitch_to_index[chord]] = 1

    measures = [initial_measure.to(DEVICE)]
    for i in range(generated_length):
        measure = model.generator(
            torch.normal(0, 1, (1, NOISE_DIMENSION), device=DEVICE),
            torch.unsqueeze(chords_tensor[i], 0),
            torch.unsqueeze(measures[i], 0),
        )
        measures.append(model_result_to_tensor(torch.squeeze(measure)))

    return torch.concat(tuple(measures), dim=1)

