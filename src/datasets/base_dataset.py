import json
import string

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset):
    def __init__(self, 
                 json_path, 
                 text_encoder,
                 transforms=None):
        
        alphabet = list(string.ascii_lowercase + " ")

        vocab = ["<blank>"] + alphabet
        
        self.char2idx = {
        char: idx 
        for idx, char in enumerate(vocab)}
        
        # print("VOCAB_SIZE", len(self.char2idx))
        self.text_encoder = text_encoder
        self.transforms = transforms
        with open(json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f) # obj: Dict(path_audio, text, audio_length)
 

    def __getitem__(self, index):
        data_dict = self.json_data[index]

        text = data_dict["text"]
        audio_path = data_dict["path"]
        audio_len = data_dict["audio_len"]

        audio = self.get_audio(audio_path)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = spectrogram.squeeze(0).transpose(0, 1)

        text_encoded = torch.tensor(
        self.text_encoder.encode(text),
        dtype=torch.long
        )
        # print("TEXT: ", text)
        # print(self._encode_text(text))
        # print("SDSA")
        # print(self.text_encoder.encode(text))
        data_obj = {
            "text": text,
            "text_encode": text_encoded,
            "spectrogram": spectrogram,
            "audio_len": audio_len
        }
        return data_obj

    
    def __len__(self):
        return len(self.json_data)
    

    def get_spectrogram(self, audio):
        mel_spectrogram = self.transforms(audio)
        
        return mel_spectrogram

    def get_audio(self, path_audio):
        waveform, sample_rate = torchaudio.load(path_audio)    
        return waveform

    def _encode_text(self, text):
        tokens = [self.char2idx[char] for char in text if char in self.char2idx]
        return tokens    
