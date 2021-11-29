import torch


class Config:
    def __init__(self):
        # device
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataset configuration
        self.N_BATCH = 32

        # network configuration
        self.N_CLASS = 10
        self.N_IMAGE = 28 * 28
        self.N_NOISE = 128

        # training configuration
        self.LR = 1e-3
        self.N_EPOCH = 200
