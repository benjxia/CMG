import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel, kernel_size, stride=1, padding=None):
        super(ResidualBlock, self).__init__()
        if padding is None:
            padding = 1
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.ln1 = nn.InstanceNorm2d(channel, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv1(x)
        logits = F.relu(logits)
        logits = self.conv2(logits)
        logits = self.ln1(logits)
        logits = logits + x
        logits = F.relu(logits)
        return logits


class ResidualBlock(nn.Module):
    def __init__(self, channel, kernel_size, stride=1, padding=None):
        super(ResidualBlock, self).__init__()
        if padding is None:
            padding = 1
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.ln1 = nn.InstanceNorm2d(channel, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv1(x)
        logits = F.relu(logits)
        logits = self.conv2(logits)
        logits = self.ln1(logits)
        logits = logits + x
        logits = F.relu(logits)
        return logits


class Encoder(nn.Module):
    def __init__(self, latent_channels=10, latent_dim=100, input_size=(2, 128, 400)):
        super(Encoder, self).__init__()
        self.latent_dim = latent_channels
        self.input_size = input_size
        self.conv0 = nn.Conv2d(2, latent_channels, kernel_size=3)
        self.conv1 = nn.ParameterList([nn.Conv2d(latent_channels, latent_channels, 5) for _ in range(20)])
        self.resd1 = nn.ParameterList([ResidualBlock(latent_channels, 3) for _ in range(10)])
        self.conv2 = nn.ParameterList([nn.Conv2d(latent_channels, latent_channels, 3) for _ in range(20)])
        # to latent dim (20, 11, 79)
        self.conv3 = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(10 * 6 * 278, 2 * latent_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.conv0(x)
        logit = F.relu(logit)

        for layer in self.conv1:
            logit = layer(logit)
            logit = F.relu(logit)

        for resblock in self.resd1:
            logit = resblock(logit)
            logit = F.relu(logit)

        for layer in self.conv2:
            logit = layer(logit)
            logit = F.relu(logit)

        logit = self.conv3(logit)
        logit = F.relu(logit)

        logit = self.fc1(logit.reshape(logit.size()[0], -1))
        return logit


class Decoder(nn.Module):
    def __init__(self, latent_channels=10, latent_dim=100, input_size=(2, 128, 400)):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, 16680)
        self.conv1 = nn.Conv2d(latent_channels, latent_channels, 3, 1, padding=1)
        self.conv2 = nn.ParameterList([nn.ConvTranspose2d(latent_channels, latent_channels, 3, 1) for _ in range(20)])
        self.resd1 = nn.ParameterList([ResidualBlock(latent_channels, 3) for _ in range(10)])
        self.conv3 = nn.ParameterList([nn.ConvTranspose2d(latent_channels, latent_channels, 5) for _ in range(20)])
        self.conv0 = nn.ConvTranspose2d(latent_channels, 2, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.fc1(x)
        logit = F.relu(logit)
        logit = logit.reshape(out.size()[0], self.latent_channels, 6, 278)
        logit = self.conv1(logit)
        logit = F.relu(logit)
        for resblock in self.conv2:
            logit = resblock(logit)
            logit = F.relu(logit)
        for resblock in self.resd1:
            logit = resblock(logit)
            logit = F.relu(logit)
        for resblock in self.conv3:
            logit = resblock(logit)
            logit = F.relu(logit)
        logit = self.conv0(logit)
        return logit

class AudioVAE(nn.Module):
    def __init__(self, latent_channels=10, input_size=(2, 128, 400)):
        super(AudioVAE, self).__init__()
        self.input_size = input_size


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
