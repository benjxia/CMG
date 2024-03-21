import torch
from AudioVAE import *
from Dataset import *
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
from torchmetrics.aggregation import RunningMean
import numpy as np
from tqdm import tqdm


BASE_DIR = './data/maestro-v2.0.0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = AudioDataset(BASE_DIR)
dataloader = DataLoader(dataset=dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=4)
transform = transforms.MelSpectrogram(sample_rate=dataset.sample_rate)

model = AudioVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

iter = tqdm(enumerate(dataloader))
iter.set_description('Training...')
metric = RunningMean(window=5)

N_EPOCH = 10

losses = []

for epoch in range(N_EPOCH):
    epoch_loss = []
    for i, batch in iter:
        batch = transform(batch).to(device)
        reconstruction, mu, log_var = model(batch)
        loss = ELBO_loss(reconstruction, batch, mu, log_var)
        epoch_loss.append(loss.item())
        iter.set_description(f"Current Loss: {metric(loss.item())}\tRunning Loss: {metric.compute()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(np.mean(epoch_loss))
    torch.save(model.state_dict(), f'audio_vae_{epoch}.pth')

