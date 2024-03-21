import torch
from torch.utils.data import Dataset
import torchaudio

import pandas as pd

import os

import random


class AudioDataset(Dataset):
    def __init__(self, path: str, sample_length=9.99, sample_rate=8000):
        super().__init__()
        self.path = path
        self.df = pd.read_csv(os.path.join(path, 'maestro-v2.0.0.csv'))
        self.sample_rate = sample_rate
        self.sample_length_frames = int(sample_length * sample_rate)  # Length of the audio segment to sample in frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        filename = self.df.iloc[index]['audio_filename']
        filename = filename[:-3] + "mp3"
        path = os.path.join(self.path, filename)

        # Load the entire audio file
        audio, original_sample_rate = torchaudio.load(path, normalize=True)

        # Resample audio if necessary
        if original_sample_rate != self.sample_rate:
            transform = torchaudio.transforms.Resample(original_sample_rate, self.sample_rate)
            audio = transform(audio)

        # Ensure audio length is at least as long as the desired segment length
        if audio.size(1) < self.sample_length_frames:
            padding = torch.zeros(audio.size(0), self.sample_length_frames - audio.size(1))
            audio = torch.cat([audio, padding], dim=1)

        # Randomly sample a segment from the audio
        if audio.size(1) > self.sample_length_frames:
            start_frame = random.randint(0, audio.size(1) - self.sample_length_frames)
            audio = audio[:, start_frame:start_frame + self.sample_length_frames]

        return audio
