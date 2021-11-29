import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, n_output),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_input):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # for true, false discrimination
        )

    def forward(self, x):
        x = self.net(x)
        return x
