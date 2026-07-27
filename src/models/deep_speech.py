import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DeepSpeech(nn.Module):
    def __init__(self,hidden_size=258, num_layers=2, dropout=0.2):
        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20)
        )

        self.input_size = 1280

        self.rnn_layers = nn.LSTM(
            input_size=self.input_size,
            hidden_size=258,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )

        self.fc = nn.Linear(516, 28)

    def forward(self, spectrogram, spectrogram_length):
        x = spectrogram.unsqueeze(1)
        x = self.conv_layers(x)
        B, C, T, F = x.shape 

        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        self.input_size = C * F
        new_spectrogram_length = self._transform_input_lengths(spectrogram_length)

        packed = pack_padded_sequence(
                x,
                lengths=new_spectrogram_length,
                batch_first=True,
                enforce_sorted=False,
        )
        
        x, _ = self.rnn_layers(packed)

        padded, lengths = pad_packed_sequence(
            x,
        batch_first=True
        )
        x = self.fc(padded)

        log_probs = nn.functional.log_softmax(x, dim=-1)

        return {
            "log_probs": log_probs.transpose(0, 1),
            "log_probs_length": lengths,
        }

    def _transform_input_lengths(self, input_lengths):
        def conv_output_length(length, kernel, stride, padding, dilation=1):
            return (
                (length + 2 * padding - dilation * (kernel - 1) - 1)
                // stride
                + 1
            )

        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                input_lengths = conv_output_length(
                    input_lengths,
                    kernel=layer.kernel_size[0],
                    stride=layer.stride[0],
                    padding=layer.padding[0],
                    dilation=layer.dilation[0],
                )
        return input_lengths