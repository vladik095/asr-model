import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_feats=80,
        hidden_size=512,
        num_layers=3,
        num_classes=28,
    ):
        super().__init__()

        ####################
        # Convolution
        ####################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),

            nn.Conv2d(
                32,
                32,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        )

        ####################
        # Calculate feature size after conv
        ####################

        freq = n_feats

        freq = self._conv_out(
            freq,
            kernel=11,
            stride=2,
            padding=5,
        )

        freq = self._conv_out(
            freq,
            kernel=11,
            stride=1,
            padding=5,
        )

        rnn_input_size = freq * 32

        print("RNN input size =", rnn_input_size)

        ####################
        # Recurrent layers
        ####################

        self.rnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):

            input_size = (
                rnn_input_size
                if i == 0
                else hidden_size * 2
            )

            self.rnns.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            self.batch_norms.append(
                nn.BatchNorm1d(hidden_size * 2)
            )

        ####################
        # Classifier
        ####################

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Hardtanh(0, 20),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, spectrogram, spectrogram_length):

        x = spectrogram.unsqueeze(1)

        x = self.conv_layers(x)

        B, C, T, F = x.shape

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, T, C * F)

        lengths = self._transform_input_lengths(
            spectrogram_length
        )

        ####################
        # RNN stack
        ####################

        for rnn, bn in zip(self.rnns, self.batch_norms):

            packed = pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            packed, _ = rnn(packed)

            x, lengths = pad_packed_sequence(
                packed,
                batch_first=True,
            )

            x = bn(
                x.reshape(-1, x.size(-1))
            ).reshape(
                x.size(0),
                x.size(1),
                x.size(2),
            )

        ####################
        # FC
        ####################

        x = self.classifier(x)

        log_probs = nn.functional.log_softmax(x, dim=-1,)

        return {
            "log_probs": log_probs.transpose(0, 1),
            "log_probs_length": lengths,
        }

    def _conv_out(
        self,
        size,
        kernel,
        stride,
        padding,
        dilation=1,
    ):
        return (
            (size + 2 * padding - dilation * (kernel - 1) - 1)
            // stride
            + 1
        )

    def _transform_input_lengths(
        self,
        input_lengths,
    ):

        for layer in self.conv_layers:

            if isinstance(layer, nn.Conv2d):

                input_lengths = (
                    (
                        input_lengths
                        + 2 * layer.padding[0]
                        - layer.dilation[0]
                        * (layer.kernel_size[0] - 1)
                        - 1
                    )
                    // layer.stride[0]
                    + 1
                )

        return input_lengths