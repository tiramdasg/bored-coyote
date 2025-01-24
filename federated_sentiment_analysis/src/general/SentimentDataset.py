import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, data, device, labels=None):
        self.device = device
        self.labels = labels
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        embedding = torch.tensor(self.data[idx], device=self.device, dtype=torch.float32).unsqueeze(0)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)
            return embedding, label
        return embedding

