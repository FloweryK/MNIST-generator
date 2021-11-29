import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from config import Config
from net import Discriminator, Generator
from util import savefig


def main():
    # configurations
    cfg = Config()

    # dataset preparation
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # dataloader preparation
    trainloader = DataLoader(dataset=trainset, batch_size=cfg.N_BATCH, shuffle=True, num_workers=4)
    testloader = DataLoader(dataset=testset, batch_size=cfg.N_BATCH, shuffle=True, num_workers=4)

    # network preparation
    G = Generator(n_input=cfg.N_NOISE + cfg.N_CLASS, n_output=cfg.N_IMAGE)
    D = Discriminator(n_input=cfg.N_IMAGE + cfg.N_CLASS)
    G.to(cfg.DEVICE)
    D.to(cfg.DEVICE)

    # losses and optimizers
    criterion = nn.BCELoss()
    G_optim = Adam(G.parameters(), lr=cfg.LR)
    D_optim = Adam(D.parameters(), lr=cfg.LR)

    for epoch in range(cfg.N_EPOCH):
        for i, (images, labels) in enumerate(trainloader):
            # get data
            images = images.view(-1, cfg.N_IMAGE)
            labels = F.one_hot(labels, cfg.N_CLASS)
            noises = torch.randn(cfg.N_BATCH, cfg.N_NOISE)

            # move data to cfg.DEVICE
            images = images.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            noises = noises.to(cfg.DEVICE)

            # merge images and labels
            images = torch.cat((images, labels), 1)  # to inform the discriminator that each image has a name
            noises = torch.cat((noises, labels), 1)  # to inform the generator which number to generate

            # check if the image is fake or not
            fakes = G(noises)
            fakes = torch.cat((G(noises), labels), 1)
            D_real = D(images)
            D_fake = D(fakes)

            # calculate the discriminator's loss and update
            D_real_loss = criterion(D_real.squeeze(1), torch.ones(cfg.N_BATCH).to(cfg.DEVICE))
            D_fake_loss = criterion(D_fake.squeeze(1), torch.zeros(cfg.N_BATCH).to(cfg.DEVICE))
            D_loss = D_real_loss + D_fake_loss
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            # check if the fakes deceived or not
            fakes = G(noises)
            fakes = torch.cat((G(noises), labels), 1)
            D_fake = D(fakes)

            # calculate the generator's loss and update
            G_loss = criterion(D_fake.squeeze(1), torch.ones(cfg.N_BATCH).to(cfg.DEVICE))
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            # Print the process
            if (i + 1) % 100 == 0:
                text = f"epoch={epoch:3d}, i={i+1:4d}, "
                text += f"D_loss={D_loss.item():.2f}, G_loss={G_loss.item():.2f}, "
                text += f"acc(real): {D_real.data.mean().item():.2f}, acc(fake): {1-D_fake.data.mean().item():.2f}"
                print(text)

        # Let's see how the digits are generated
        test_noises = torch.randn(10, cfg.N_NOISE)
        test_noises = torch.cat((test_noises, torch.eye(10)), 1)
        test_noises = test_noises.to(cfg.DEVICE)
        samples = G(test_noises)
        savefig(epoch=epoch, samples=samples)


if __name__ == "__main__":
    main()
