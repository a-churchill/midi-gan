import torch
import torch.nn as nn
import torch.nn.functional as F

from discriminator import Discriminator
from generator import Generator
from parameters import ADAM_BETA, NOISE_DIMENSION


def init_weights(model):
    """Custom weight initialization for generator and discriminator. Code from
    [Pytorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def prediction_loss(prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
    """Computes loss for a set of predictions (single-element tensors)."""
    labels = torch.tensor(
        [1.0 if is_real else 0.0], device=prediction.device
    ).expand_as(prediction)
    loss = F.mse_loss(prediction, labels)
    return loss


def set_requires_grad(net: nn.Module, requires_grad=False):
    """Saves computation by avoiding computing gradient unless necessary. Credit to
    [this repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for
    implementation inspiration.

    Args:
        nets (List[nn.Module]): nets to update
        requires_grad (bool, optional): whether `nets` require gradient. Defaults to False.
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


class GAN(nn.Module):
    """Generative adversarial network for writing music."""

    def __init__(
        self,
        learning_rate=0.0002,
        generator_conv_channels=128,
        discriminator_conv_channels=64,
        lambda_1=1.0,
        lambda_2=0.01,
        adam_param_1=0.5,
        adam_param_2=0.999,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.adam_param_1 = adam_param_1
        self.adam_param_2 = adam_param_2

        self.generator = Generator(conv_channels=generator_conv_channels)
        self.generator.apply(init_weights)
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=(ADAM_BETA, 0.999)
        )

        self.discriminator = Discriminator(conv_channels=discriminator_conv_channels)
        self.discriminator.apply(init_weights)
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.adam_param_1, self.adam_param_2),
        )

        # original bars, from input
        self.real_bars = None
        self.last_measure_bars = None

        # "creative" input
        self.creative_input = None

        # generated bars
        self.fake_bars = None

        # losses, for reporting
        self.fake_loss = None
        self.real_loss = None
        self.gan_loss = None
        self.feature_map_loss = None
        self.result_loss = None

    def forward(
        self,
        real_bars: torch.Tensor,
        last_measure_bars: torch.Tensor,
        creative_input: torch.Tensor,
    ) -> torch.Tensor:
        """Generates fake music"""
        self.real_bars = real_bars
        self.last_measure_bars = last_measure_bars
        self.creative_input = creative_input

        self.fake_bars = self.generator(
            torch.normal(
                0,
                1,
                (self.real_bars.size(0), NOISE_DIMENSION),
                device=self.real_bars.device,
            ),
            self.creative_input,
            self.last_measure_bars,
        )
        return self.fake_bars

    def backward(self):
        """Update generator and discriminator weights"""
        # make sure forward already ran
        if self.fake_bars is None:
            raise Exception(
                "fake_output not generated, make sure to run forward() before backward()!"
            )

        # update discriminator
        set_requires_grad(self.discriminator, True)
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

        # update generator
        set_requires_grad(self.discriminator, False)
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # update generator again (recommended by midinet paper)
        # self.forward(
        #     self.real_bars, self.last_measure_bars, self.creative_input,
        # )

        # self.optimizer_g.zero_grad()
        # self.backward_g()
        # self.optimizer_g.step()

        # reset input until next forward call
        self.real_bars = None
        self.fake_bars = None

    def backward_d(self):
        """Computes loss and gradients for discriminator network"""
        # run discriminator (with detached input, to make sure generator is not affected)
        output_d_fake, _ = self.discriminator(
            self.fake_bars.detach(), self.creative_input.detach()
        )

        # compute loss for fake batch
        self.fake_loss = prediction_loss(output_d_fake, False)

        # run discriminator (now with real bars)
        output_d_real, _ = self.discriminator(self.real_bars, self.creative_input)

        # compute loss for real batch
        self.real_loss = prediction_loss(output_d_real, True)

        # combine loss, compute gradients
        loss = (self.fake_loss + self.real_loss) * 0.5
        loss.backward()

    def backward_g(self):
        """Computes loss and gradients for generator network"""
        # run loss on discriminator results, pretending they are real
        output_d_fake, feature_map_fake = self.discriminator(
            self.fake_bars, self.creative_input.detach()
        )
        self.gan_loss = prediction_loss(output_d_fake, True)

        # compute feature maps for actual input
        _, feature_map_real = self.discriminator(self.real_bars, self.creative_input)

        self.feature_map_loss = (
            F.mse_loss(feature_map_fake, feature_map_real, reduction="mean")
            * self.lambda_1
        )
        self.result_loss = (
            F.mse_loss(self.fake_bars, self.real_bars, reduction="mean") * self.lambda_2
        )

        # compute gradients
        loss = self.gan_loss + self.feature_map_loss + self.result_loss
        loss.backward()

    def report_losses(self):
        """Logs the current losses"""
        print("Current loss values:\n")
        print(f"\treal: {self.real_loss}")
        print(f"\tfake: {self.fake_loss}")
        print(f"\tgan: {self.gan_loss}")
        print(f"\tfeature map: {self.feature_map_loss}")
        print(f"\tresult: {self.result_loss}")
        return [
            self.real_loss.item(),
            self.fake_loss.item(),
            self.gan_loss.item(),
            self.feature_map_loss.item(),
            self.result_loss.item(),
        ]
