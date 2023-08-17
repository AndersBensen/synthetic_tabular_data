import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 16),
            # nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Linear(16, 32),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32, data_dim),
        )
        
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):

    def __init__(self, data_dim):
        print(f" data dim {data_dim}")
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(data_dim, 32),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    