import torch
import torch.nn as nn

from parameters import (
    BAR_LENGTH,
    MIDI_NOTES,
    NOISE_DIMENSION,
    CREATIVE_DIMENSION,
)
from utils import concat_creative_input


class Generator(nn.Module):
    """Generates music"""

    def __init__(self, conv_channels=128) -> None:
        super().__init__()
        self.conv_channels = conv_channels

        # conditioner
        self.conditioner_1 = nn.Sequential(
            nn.Conv2d(
                1, self.conv_channels, kernel_size=(MIDI_NOTES, 1), stride=(2, 1)
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(0.2),
        )

        self.conditioner_2 = nn.Sequential(
            nn.Conv2d(
                self.conv_channels, self.conv_channels, kernel_size=(1, 2), stride=2
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(0.2),
        )

        self.conditioner_3 = nn.Sequential(
            nn.Conv2d(
                self.conv_channels, self.conv_channels, kernel_size=(1, 2), stride=2
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(0.2),
        )

        self.conditioner_4 = nn.Sequential(
            nn.Conv2d(
                self.conv_channels, self.conv_channels, kernel_size=(1, 2), stride=2
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(0.2),
        )

        # generator
        self.noise_1 = nn.Sequential(
            nn.Linear(NOISE_DIMENSION + CREATIVE_DIMENSION, self.conv_channels * 16),
            nn.BatchNorm1d(self.conv_channels * 16),
            nn.ReLU(),
        )
        self.noise_2 = nn.Sequential(
            nn.Linear(
                self.conv_channels * 16 + CREATIVE_DIMENSION, self.conv_channels * 4
            ),
            nn.BatchNorm1d(self.conv_channels * 4),
            nn.ReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.conv_channels * 2 + CREATIVE_DIMENSION + self.conv_channels,
                self.conv_channels,
                kernel_size=(1, 2),
                stride=2,
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.conv_channels + CREATIVE_DIMENSION + self.conv_channels,
                self.conv_channels,
                kernel_size=(1, 2),
                stride=2,
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.ConvTranspose2d(
                self.conv_channels + CREATIVE_DIMENSION + self.conv_channels,
                self.conv_channels,
                kernel_size=(1, 2),
                stride=2,
            ),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(),
        )
        self.conv_4 = nn.Sequential(
            nn.ConvTranspose2d(
                self.conv_channels + CREATIVE_DIMENSION + self.conv_channels,
                1,
                kernel_size=(MIDI_NOTES, 1),
                stride=1,
            ),
        )

    def forward(
        self, z: torch.Tensor, y: torch.Tensor, prev_bar: torch.Tensor
    ) -> torch.Tensor:
        """Generates a bar of MIDI music.

        Args:
            z (torch.Tensor): random noise (BATCH_SIZE x NOISE_DIMENSION)
            y (torch.Tensor): "creative" input (BATCH_SIZE x CREATIVE_DIMENSION)
            prev_bar (torch.Tensor): the generated previous bar (BATCH_SIZE x MIDI_NOTES x BAR_LENGTH)

        Returns:
            torch.Tensor: a bar of generated music
        """

        # for internal representation, within the network: dimensions are B x C x H x W
        # B = batch
        # C = channels
        # H = height (MIDI_NOTES = 128)
        # W = width (BAR_LENGTH = 16)

        # when we concat things, we want to do it along the channels dimension

        # note that the comments explaining the dimensions of things will ignore the batch dimension

        # batch prev_bar properly (input is B x H x W, we need B x C x H x W)
        prev_bar = prev_bar.reshape((-1, 1, MIDI_NOTES, BAR_LENGTH))

        y_batched = y.reshape((-1, CREATIVE_DIMENSION, 1, 1))

        # compute conditioner outputs first
        c1 = self.conditioner_1(prev_bar)  # self.conv_channels x 1 x BAR_LENGTH
        c2 = self.conditioner_2(c1)  # self.conv_channels x 1 x (BAR_LENGTH / 2)
        c3 = self.conditioner_3(c2)  # self.conv_channels x 1 x (BAR_LENGTH / 4)
        c4 = self.conditioner_4(c3)  # self.conv_channels x 1 x (BAR_LENGTH / 8)

        # fully connected layers for random noise and "creative" input
        result: torch.Tensor = torch.concat((z, y), dim=1)
        # NOISE_DIMENSION + CREATIVE_DIMENSION

        result = self.noise_1(result)
        # self.conv_channels * 16

        result = torch.concat((result, y), dim=1)
        # self.conv_channels * 16 + CREATIVE_DIMENSION

        result = self.noise_2(result)
        # self.conv_channels * 4

        # reshape for convolutional layers

        result = result.reshape((-1, self.conv_channels * 2, 1, 2))
        # (self.conv_channels * 2) x 1 x 2

        # convolutional layers

        result = concat_creative_input(result, y_batched)
        # (self.conv_channels * 2 + CREATIVE_DIMENSION) x 1 x (BAR_LENGTH / 8)

        result = torch.concat((result, c4), dim=1)
        # (self.conv_channels * 2 + CREATIVE_DIMENSION + self.conv_channels) x 1 x (BAR_LENGTH / 8)

        result = self.conv_1(result)
        # self.conv_channels x 1 x (BAR_LENGTH / 4)

        result = concat_creative_input(result, y_batched)
        # (self.conv_channels + CREATIVE_DIMENSION) x 1 x (BAR_LENGTH / 4)

        result = torch.concat((result, c3), dim=1)
        # (self.conv_channels + CREATIVE_DIMENSION + self.conv_channels) x 1 x (BAR_LENGTH / 4)

        result = self.conv_2(result)
        # self.conv_channels x 1 x (BAR_LENGTH / 2)

        result = concat_creative_input(result, y_batched)
        # (self.conv_channels + CREATIVE_DIMENSION) x 1 x (BAR_LENGTH / 2)

        result = torch.concat((result, c2), dim=1)
        # (self.conv_channels + CREATIVE_DIMENSION + self.conv_channels) x 1 x (BAR_LENGTH / 2)

        result = self.conv_3(result)
        # self.conv_channels x 1 x BAR_LENGTH

        result = concat_creative_input(result, y_batched)
        # (self.conv_channels + CREATIVE_DIMENSION) x 1 x BAR_LENGTH

        result = torch.concat((result, c1), dim=1)
        # (self.conv_channels + CREATIVE_DIMENSION + self.conv_channels) x 1 x BAR_LENGTH

        result = self.conv_4(result)
        # 1 x MIDI_NOTES x BAR_LENGTH

        # final result reshaping
        result = result.reshape((-1, MIDI_NOTES, BAR_LENGTH))

        return result
