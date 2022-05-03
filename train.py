import os
from typing import Union

import torch
from ray import tune
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from debug import plot_midi
from gan import GAN
from parameters import DEVICE, NUM_EPOCHS
from run_model import run_model


def train(
    model: GAN, dataloader: DataLoader, writer: Union[SummaryWriter, None] = None,
):
    """Trains `model` with the given data."""
    is_tune = writer is None
    model = model.to(DEVICE)
    model.train()

    for epoch in range(NUM_EPOCHS):
        if not is_tune:
            print("\nEPOCH", epoch + 1, "of", NUM_EPOCHS)

        model.train()
        data_iter = dataloader if is_tune else tqdm(dataloader)

        for last_measure, current_measure, chord_tensor in data_iter:
            if current_measure.size(0) == 1:
                # batch norm doesn't work with batch size = 1
                continue

            last_measure = last_measure.to(DEVICE)
            current_measure = current_measure.to(DEVICE)
            chord_tensor = chord_tensor.to(DEVICE)

            model.forward(
                real_bars=current_measure,
                last_measure_bars=last_measure,
                creative_input=chord_tensor,
            )
            model.backward()

        if is_tune:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (
                        model.state_dict(),
                        model.optimizer_d.state_dict(),
                        model.optimizer_g.state_dict(),
                    ),
                    path,
                )

            tune.report(loss=model.gan_loss.item(), real_loss=model.real_loss.item())

        else:
            writer.add_scalars(
                "Loss",
                {
                    "fake loss": model.fake_loss,
                    "real loss": model.real_loss,
                    "gan loss": model.gan_loss,
                    "feature map loss": model.feature_map_loss,
                    "result loss": model.result_loss,
                },
                epoch,
            )

            model.eval()
            sample = run_model(
                model, current_measure[0], ["C", "E", "G", "F", "D", "E", "C"]
            )
            writer.add_figure("Sample", plot_midi(sample), epoch)
            writer.flush()
