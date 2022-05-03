import torch
import torch.nn as nn

from parameters import (
    BAR_LENGTH,
    CREATIVE_DIMENSION,
    MIDI_NOTES,
)
from utils import concat_creative_input


class Discriminator(nn.Module):
    """Determines whether a bar is real or fake."""

    def __init__(self, conv_channels=64) -> None:
        super().__init__()

        self.conv_channels = conv_channels

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                1 + CREATIVE_DIMENSION,
                self.conv_channels,
                kernel_size=(MIDI_NOTES, 2),
                stride=2,
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(0.2),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                self.conv_channels + CREATIVE_DIMENSION,
                self.conv_channels,
                kernel_size=(1, 2),
                stride=2,
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(0.2),
        )

        self.linear_1 = nn.Sequential(
            nn.Linear(
                self.conv_channels * int(BAR_LENGTH / 4) + CREATIVE_DIMENSION,
                self.conv_channels,
            ),
            nn.BatchNorm1d(self.conv_channels),
            nn.LeakyReLU(0.2),
        )

        self.linear_2 = nn.Linear(self.conv_channels + CREATIVE_DIMENSION, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Determines if a bar of music is real or fake.

        Args:
            x (torch.Tensor): batched bars of music (BATCH_SIZE x MIDI_NOTES x BAR_LENGTH)
            y (torch.Tensor): batched "creative" input (BATCH_SIZE x CREATIVE_DIMENSION)

        Returns:
            torch.Tensor: probability each bar is real (BATCH_SIZE x 1)
        """
        # for internal representation, within the network: dimensions are B x C x H x W
        # B = batch
        # C = channels
        # H = height (MIDI_NOTES = 128)
        # W = width (BAR_LENGTH = 16)

        # when we concat things, we want to do it along the channels dimension

        # note that the comments explaining the dimensions of things will ignore the batch dimension

        # reshape to add channel dimension
        x = x.reshape((-1, 1, MIDI_NOTES, BAR_LENGTH))

        y_batched = y.reshape((-1, CREATIVE_DIMENSION, 1, 1))

        result: torch.Tensor = concat_creative_input(x, y_batched)
        # (1 + CREATIVE DIMENSION) x MIDI_NOTES x BAR_LENGTH

        result = self.conv_1(result)
        feature_map = result
        # self.conv_channels x 1 x (BAR_LENGTH / 2)

        result = concat_creative_input(result, y_batched)
        # (self.conv_channels + CREATIVE_DIMENSION) x 1 x (BAR_LENGTH / 2)

        result = self.conv_2(result)
        # self.conv_channels x 1 x (BAR_LENGTH / 4)

        result = result.reshape((-1, self.conv_channels * int(BAR_LENGTH / 4)))
        # self.conv_channels * BAR_LENGTH / 4

        result = torch.concat((result, y), dim=1)
        # self.conv_channels * BAR_LENGTH / 4 + CREATIVE_DIMENSION

        result = self.linear_1(result)
        # self.conv_channels

        result = torch.concat((result, y), dim=1)
        # self.conv_channels + CREATIVE_DIMENSION

        result = self.linear_2(result)
        # 1

        return result, feature_map
