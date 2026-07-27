import torch
import torchaudio


class LogMelSpectrogram:
    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

    def __call__(self, data):
        # data: waveform [T]

        spec = self.mel(data)

        # log scale
        spec = torch.log(spec + 1e-9)

        return spec