import torch
from torch import nn


class BaselineModelRNN(nn.Module):
    def __init__(
        self,
        n_feats,
        n_tokens,
        hidden_size=256,
        num_layers=2,
        dropout=0.15,
    ):
        super().__init__()

        self.prenet = nn.Sequential(
            nn.Linear(n_feats, hidden_size),
            nn.ReLU(),
        )

        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_tokens),
        )

    def forward(self, x):
        # x = batch["spectrograms"]          # (Batch, Time, Freqs)
        print("SHAPE ", x.unsqueeze(1).shape)
        print(x[0])
        x = self.prenet(x)

        x, _ = self.encoder(x)

        x = self.classifier(x)

        log_probs = nn.functional.log_softmax(x, dim=-1)

        return {
            "log_probs": log_probs.transpose(0, 1),
        }

    def transform_input_lengths(self, input_lengths):
        return input_lengths