from models import Generator, Discriminator
from data import get_dataset
from training import train

from torch.utils.data import DataLoader
import torch

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # X, X_mean, X_std, columns = get_dataset("../datasets/diabetes.csv")
    # X = X[:,:-1]
    X, X_mean, X_std, columns, df = get_dataset("../datasets/winequality-red.csv")
    X = (X - X_mean) / X_std
    print(f"Dataset shape: {X.shape}")

    latent_dim = 3

    generator = Generator(latent_dim=latent_dim, data_dim=X.shape[1])
    discriminator = Discriminator(data_dim=X.shape[1])

    lr = 3e-4
    beta1 = 0.5

    optimizer_g = torch.optim.Adam(generator.parameters(), lr, betas=(beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr, betas=(beta1, 0.999))
    
    criterion = torch.nn.BCELoss()

    epochs = 3700
    batch_size = 128

    input = torch.from_numpy(X).float()
    dataloader = DataLoader(input, batch_size=batch_size, shuffle=True, drop_last=True)

    discriminator_loss, generator_loss = train(
                generator=generator, discriminator=discriminator, optimizer_d=optimizer_d, optimizer_g=optimizer_g, 
                epochs=epochs, criterion=criterion, batch_size=batch_size, dataloader=dataloader, latent_dim=latent_dim
    )

    plt.plot(discriminator_loss, label="D Loss")
    plt.plot(generator_loss, label="G Loss")
    plt.legend()
    plt.show()
