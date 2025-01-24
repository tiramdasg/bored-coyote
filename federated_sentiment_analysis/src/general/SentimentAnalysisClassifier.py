import logging
import time

import numpy as np
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from general.SentimentDataset import SentimentDataset


class SentimentAnalysisClassifier(nn.Module):
    """
    Input to this model is a tfidf vectorizer - vector.
    """

    def __init__(self, input_length, model_path, optimizer = None):
        super(SentimentAnalysisClassifier, self).__init__()
        self.model_path = model_path
        self.input_length = input_length
        self.decision_threshold = 0.5
        self.optimizer = optimizer
        self.best_loss = np.inf
        self.conv1d = nn.Conv1d(self.input_length, 32, kernel_size=3, padding='same')

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc_sequence = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.dropout = nn.Dropout(0.4)
        self.fc_proj = nn.Linear(7500, 256)
        self.lstm = nn.LSTM(256, 256, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = F.relu(self.conv1d(x))
        x = self.pool(x)
        x = self.fc_proj(x)
        x, _ = self.lstm(x)  # lstm expects input of shape (batch_size, seq_length, input_size)
        x = x[:, -1, :]  # Take the last output of the LSTM
        x = self.dropout(x)
        return F.sigmoid(self.fc_sequence(x))

    def fit(self, num_epochs, data, labels, device, verbose=0, batch_size=512, lr=0.001):
        """Datapoint embeddings are TfidfVectorizer output with max_features=15000 (call toarray() on it) and np.float32 over reddit + twitter
        sentiment analysis dateset.
        proba controls, whether class probability for True is returned, or class."""
        self.to(device)
        # dataset and loader initializations
        ds = SentimentDataset(data=data, device=device, labels=labels)

        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # loss function and optimizer
        criterion = nn.BCELoss()
        if not self.optimizer:
            optimizer = optim.Adam(self.parameters(), lr=lr)
            self.optimizer = optimizer
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.optimizer.param_groups[0]['lr'])

        # initializing avg loss for reporting
        avg_loss = 0.0

        best_loss = self.best_loss

        # actual train loop:
        for epoch in range(num_epochs):
            self.train()
            start_time = time.time()
            running_loss = 0.0
            running_acc = 0.0

            # for batch in dataloader compute loss and backwards etc.
            for embeddings, labels in dataloader:
                optimizer.zero_grad()  # idk, just a best practice

                outputs = self(embeddings)

                loss = criterion(outputs, labels.view(-1, 1).float())
                preds = (outputs >= 0.5).int()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            end_time = time.time()
            avg_loss = running_loss / len(dataloader)


            if verbose:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Duration: [{int(end_time - start_time)}s], Loss: [{avg_loss:.4f}]")
            if avg_loss < best_loss:
                best_weights = self.state_dict().copy()
                model_snapshot = {
                    "loss": avg_loss,
                    "state_dict": best_weights,
                    "optim": optimizer.state_dict()
                }
                best_loss = avg_loss
                self.best_loss = best_loss
                torch.save(model_snapshot, self.model_path)

        return avg_loss

    def store(self, path=None):
        model_snapshot = {"loss": self.best_loss,
                          "state_dict": self.state_dict().copy(),
                          "optim": self.optimizer.state_dict()}
        if path:
            torch.save(model_snapshot, path)
        else:
            torch.save(model_snapshot, self.model_path)

    def load_model(self, device, path=None, lr=0.001):
        model_snap_shot = torch.load(path if path else self.model_path, map_location=device)

        self.load_state_dict(model_snap_shot["state_dict"])
        opt = optim.Adam(self.parameters(), lr)
        opt.load_state_dict(model_snap_shot["optim"])
        self.optimizer = opt
        self.best_loss = model_snap_shot["loss"]
        logging.info(f"best loss: {self.best_loss}")

    def predict(self, datapoint_embeddings, device, proba=True):
        """Datapoint's embeddings are TfidfVectorizer output with max_features=15000 (call toarray() on it)
         and np.float32 over several reddit + twitter sentiment analysis datasets.
        proba controls, whether class probability for True is returned, or class."""
        self.to(device)
        # prediction
        self.eval()

        dataset = SentimentDataset(datapoint_embeddings, device)
        dataloader = DataLoader(dataset, batch_size=512)
        outputs = []
        for embeddings in dataloader:
            with torch.no_grad():
                output = self(embeddings).squeeze(-1)
                if proba:
                    outputs.append(output.cpu().tolist())
                else:
                    outputs.append((output >= self.decision_threshold).cpu().int().tolist())

        return [round(item, 5) for sublist in outputs for item in sublist]
