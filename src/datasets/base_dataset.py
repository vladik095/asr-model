import json
import os
import string

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset):
    def __init__(self, path_data_dir, text_encoder, transforms=None):

        self.data_index = []

        self._build_data_index(path_data_dir)

        self.text_encoder = text_encoder
        self.transforms = transforms

    def __getitem__(self, index):
        data_dict = self.data_index[index]

        text = data_dict["text"]
        audio_path = data_dict["path_audio"]

        audio = self.get_audio(audio_path)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = spectrogram.squeeze(0).transpose(0, 1)

        text_encoded = torch.tensor(self.text_encoder.encode(text), dtype=torch.long)

        data_obj = {
            "text": text,
            "text_encode": text_encoded,
            "spectrogram": spectrogram,
        }
        return data_obj

    def __len__(self):
        return len(self.data_index)

    def get_spectrogram(self, audio):
        mel_spectrogram = self.transforms(audio)

        return mel_spectrogram

    def get_audio(self, path_audio):
        waveform, _ = torchaudio.load(path_audio)
        return waveform

    def _build_data_index(self, path_data_dir):
        for dirname, _, filenames in os.walk(path_data_dir):
            for filename in filenames:
                full_path = os.path.join(dirname, filename)

                if filename.endswith("trans.txt"):
                    with open(full_path, "r", encoding="utf-8") as file:
                        for line in file:
                            line = line.strip()
                            audio_id, text = line.split(maxsplit=1)

                            audio_path = (
                                os.path.dirname(full_path) + f"/{audio_id}.flac"
                            )
                            
                            exmp = {"path_audio": audio_path, "text": text.lower()}
                            self.data_index.append(exmp)
