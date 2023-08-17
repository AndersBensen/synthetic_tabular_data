import torch 
from torch import nn
import numpy as np
import random


# Get either CUDA or Mac GPU or simply CPU. 
def get_device(): 
    device = ""
    if (torch.cuda.is_available()): device = "cuda:0"
    # elif (torch.has_mps): device = "mps" # This is the MacBook M1 GPU
    else: device = "cpu"
    return device

# DCGAN paper tells to initial weights from normal distribution with mean 0 and std 0.02.
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def get_tensors_in_range(r1, r2, batch_size):
    return (r1 - r2) * torch.rand(batch_size) + r2

def get_real_labels(batch_size):
    r1 = 0.9
    r2 = 1.2
    return get_tensors_in_range(r1, r2, batch_size)

    # return torch.Tensor(batch_size).fill_(random.uniform(0.9, 1.2))

def get_fake_labels(batch_size):
    r1 = 0.0
    r2 = 0.3
    return get_tensors_in_range(r1, r2, batch_size)

    # return torch.Tensor(batch_size).fill_(random.uniform(0.0, 0.3))

# Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
def train_discriminator(discriminator, optimizer_d, criterion, real_data, fake_data, device):
    batch_size = real_data.shape[0]

    optimizer_d.zero_grad()

    # Train with real data
    # real_label = get_real_labels(batch_size).to(device).unsqueeze(1)
    real_label = torch.ones(batch_size).to(device).unsqueeze(1)
    output_real = discriminator(real_data)
    loss_real = criterion(output_real, real_label)
    loss_real.backward()

    # Train with fake data
    # fake_label = get_fake_labels(batch_size).to(device).unsqueeze(1)
    fake_label = torch.zeros(batch_size).to(device).unsqueeze(1)
    output_fake = discriminator(fake_data)
    loss_fake = criterion(output_fake, fake_label)
    loss_fake.backward()

    optimizer_d.step()

    return loss_real + loss_fake

# Update Generator: maximize log(D(G(z)))
def train_generator(discriminator, optimizer_g, criterion, fake_data, device):
    batch_size = fake_data.shape[0]
    # real_label = get_real_labels(batch_size).to(device).unsqueeze(1)
    real_label = torch.ones(batch_size).to(device).unsqueeze(1)

    optimizer_g.zero_grad()

    output = discriminator(fake_data)
    loss = criterion(output, real_label)

    loss.backward()

    optimizer_g.step()

    return loss

# Actual training loop
def train(generator, discriminator, optimizer_g, optimizer_d, epochs, criterion, dataloader, batch_size, latent_dim):
    device = get_device()
    print(f"Device; {device}")

    generator.to(device)
    generator.apply(weights_init)

    discriminator.to(device)
    discriminator.apply(weights_init)

    discriminator_loss = []
    generator_loss = []

    # Save some initial noise to see the data after each epoch
    # initial_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    print("### Beginning training ###")

    # For each epoch
    for epoch in range(epochs):
        print(f'Epoch {epoch} of {epochs}')
        # For each batch in the dataloader
        run_loss_d = 0
        run_loss_g = 0
        for i, real_data in enumerate(dataloader, 0):
            real_data = real_data.to(device)

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(noise)

            # Train discriminator
            loss_discriminator = train_discriminator(discriminator, optimizer_d, criterion, real_data, fake_data.detach(), device).item()
            run_loss_d += loss_discriminator

            # Train generator
            loss_generator = train_generator(discriminator, optimizer_g, criterion, fake_data, device).item()
            run_loss_g += loss_generator
            
            if (i % 100 == 0):
                 print(f'- Iteration {i}: Generator loss {loss_generator} -- Discriminator loss {loss_discriminator} -')

        epoch_loss_d = run_loss_d/len(dataloader)
        epoch_loss_g = run_loss_g/len(dataloader)

        discriminator_loss.append(epoch_loss_d)
        generator_loss.append(epoch_loss_g)

        print(f'--- Epoch {epoch} avg: Generator loss {epoch_loss_g} -- Discriminator loss {epoch_loss_d} ---')


    print("### Ending training ###")
    torch.save(generator.state_dict(), f"../models/generator.pt")
    torch.save(discriminator.state_dict(), f"../models/discriminator.pt")
    return discriminator_loss, generator_loss