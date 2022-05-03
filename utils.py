import torch

from parameters import CREATIVE_DIMENSION


def concat_creative_input(
    result: torch.Tensor, y_batched: torch.Tensor
) -> torch.Tensor:
    """Concatenates the "creative" input (`y`) to the current result

    Args:
        result (torch.Tensor): current result, with shape `BATCH_SIZE x CONV_CHANNELS x H x W`
        y_batched (torch.Tensor): y input, with shape `BATCH_SIZE x CREATIVE_CHANNELS x 1 x 1`

    Returns:
        torch.Tensor: tensor with `y_batched` concatenated along channel dimension
    """
    return torch.concat(
        (
            result,
            y_batched
            * torch.ones(
                (result.size(0), CREATIVE_DIMENSION, result.size(2), result.size(3)),
                device=result.device,
            ),
        ),
        dim=1,
    )
