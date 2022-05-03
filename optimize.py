import os

import torch
from data import dataloader

from gan import GAN
from train import train


def optimize_midinet(config, checkpoint_dir=None):
    """Optimizes our generative adversarial network, by reporting data to `tune`

    Args:
        config (Map): map from configs names to their values
        checkpoint_dir (_type_, optional): _description_. Defaults to None.
    """
    model = GAN(**config)
    if checkpoint_dir is not None:
        model_state, optimizer_d_state, optimizer_g_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        model.optimizer_d.load_state_dict(optimizer_d_state)
        model.optimizer_g.load_state_dict(optimizer_g_state)

    train(model, dataloader)
