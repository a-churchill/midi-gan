import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter

from data import dataloader
from debug import plot_midi, tensor_to_midi
from gan import GAN
from optimize import optimize_midinet
from parameters import NUM_EPOCHS
from run_model import run_model
from train import train

config = {
    "adam_param_1": tune.uniform(0, 1),
    "adam_param_2": tune.uniform(0, 1),
    "discriminator_conv_channels": tune.sample_from(
        lambda _: 2 ** np.random.randint(2, 9)
    ),
    "generator_conv_channels": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "lambda_1": tune.uniform(0.01, 1),
    "lambda_2": tune.uniform(0.01, 1),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
}

best_config = {
    "adam_param_1": 0.10671179043958812,
    "adam_param_2": 0.17622715813119028,
    "discriminator_conv_channels": 128,
    "generator_conv_channels": 32,
    "lambda_1": 0.585780650517357,
    "lambda_2": 0.8235284903358508,
    "learning_rate": 0.0011518093683686084,
}


def main(is_tune: bool):
    if is_tune:
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=NUM_EPOCHS,
            grace_period=1,
            reduction_factor=2,
        )
        reporter = CLIReporter(
            metric_columns=["loss", "real_loss", "training_iteration"]
        )

        result = tune.run(
            optimize_midinet,
            resources_per_trial={"cpu": 8, "gpu": 1},
            config=config,
            num_samples=30,
            scheduler=scheduler,
            progress_reporter=reporter,
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config:", best_trial.config)
        print("Best trial final validation loss:", best_trial.last_result["loss"])

        model = GAN(**best_trial.config)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, _, _ = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)

    else:
        writer = SummaryWriter("runs/midi_net_5")
        model = GAN(**best_config)
        train(model, dataloader, writer)

    _, current_measure, _ = next(iter(dataloader))

    piece = run_model(
        model,
        current_measure[0],
        [
            "C",
            "F",
            "G",
            "C",
            "D",
            "G",
            "E",
            "G",
            "D",
            "F",
            "G",
            "C",
            "D",
            "E",
            "F",
            "G",
            "C",
        ],
    )

    if is_tune:
        plot_midi(piece)
        plt.show()
    else:
        writer.add_figure("Piece", plot_midi(piece))
        writer.close()

    tensor_to_midi(piece).show()


if __name__ == "__main__":
    main(is_tune=False)
