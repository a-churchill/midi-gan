import sys

import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

# beta value for Adam optimizer
ADAM_BETA = 0.5

# number of quantized notes in 1 bar
BAR_LENGTH = 16

# batch size during training
BATCH_SIZE = 72

# dimension to use for the "creative" 1D input (y)
CREATIVE_DIMENSION = 13

# GPU device to run training on
DEVICE = torch.device("cuda:0")

# the number of possible MIDI notes (where we say 0 represents rest)
MIDI_NOTES = 128

# dimension of the input noise vector to use
NOISE_DIMENSION = 100

# number of training epochs
NUM_EPOCHS = 40
