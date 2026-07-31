import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.bn = nn.BatchNorm1d(hidden_size * 2)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x


class BestDeepSpeech(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Linear(2, 33)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        )

        rnn_input = 1280  # 32 (channel) * 40 (freq)

        self.rnn_blocks = nn.Sequential(
            GRUBlock(rnn_input, 258), GRUBlock(516, 258), GRUBlock(516, 258)
        )

        self.fc = nn.Linear(516, 28)

    def forward(self, x, lengths):
        """
        Args:
            x (torch.tensor): B T F
        """
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.conv_layer(x)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, T, C * F)

        transformed_lengths = self.transform_input_lengths(lengths)
        x = self.rnn_blocks(x)

        x = self.fc(x)
        # print("SHAPE X: ", transformed_lengths)
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)

        return {
            "log_probs": log_probs.transpose(0, 1),
            "log_probs_length": transformed_lengths,
        }

    def transform_input_lengths(self, input_lengths):
        """
        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        seq_len = input_lengths
        for m in self.conv_layer.modules():
            if isinstance(m, nn.Conv2d):
                seq_len = (
                    seq_len
                    + 2 * m.padding[0]
                    - m.dilation[0] * (m.kernel_size[0] - 1)
                    - 1
                ) // m.stride[0] + 1
        return seq_len.int()
