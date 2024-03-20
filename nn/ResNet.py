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


class Encoder(nn.Module):
    def __init__(self, latent_dim=10, input_size=(2, 128, 400)):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.conv1 = nn.Conv2d(self.input_size[0], latent_dim, 3, 1)
        self.resd1 = nn.ParameterList([ResidualBlock(latent_dim, 3) for _ in range(10)])
        self.conv2 = nn.ParameterList([nn.Conv2d(latent_dim, latent_dim, 3) for _ in range(20)])
        # to latent dim (20, 11, 79)
        self.conv3 = nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.conv1(x)
        logit = F.relu(logit)
        for resblock in self.resd1:
            logit = resblock(logit)
            logit = F.relu(logit)
        logit, idx1 = F.max_pool2d(logit, kernel_size=2, stride=2, return_indices=True)
        for layer in self.conv2:
            logit = layer(logit)
            logit = F.relu(logit)
        logit, idx2 = F.max_pool2d(logit, kernel_size=2, stride=2, return_indices=True)
        # logit = torch.flatten(logit, start_dim=1)
        # logit = self.fc1(logit)
        logit = self.conv3(logit)
        return (logit, idx1, idx2)


class Decoder(nn.Module):
    def __init__(self, latent_dim=200, input_size=(2, 128, 401)):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(latent_dim, latent_dim, 3, 1, padding=1)
        # self.fc1 = nn.Linear(latent_dim, 20 * (self.input_size[1] // 4) * (self.input_size[2] // 4))
        self.conv2 = nn.ParameterList([nn.ConvTranspose2d(latent_dim, latent_dim, 3, 1) for _ in range(20)])
        self.resd1 = nn.ParameterList([ResidualBlock(20, 3) for _ in range(10)])
        self.conv3 = nn.ConvTranspose2d(self.input_size[0], latent_dim, 3, 1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        logit, idx1, idx2 = x
        logit = self.conv1(logit)
        logit = F.max_unpool2d(logit, idx2, kernel_size=2)
        for resblock in self.conv2:
            logit = resblock(logit)
            logit = F.relu(logit)
        logit = F.max_unpool2d(logit, idx1, kernel_size=2)
        for resblock in self.resd1:
            logit = resblock(logit)
            logit = F.relu(logit)
        logit = self.conv3(logit)

        return logit
