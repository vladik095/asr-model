import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items):
    
    spectrograms = []
    lens_specs = []

    texts_enc = []
    lens_texts = []

    texts = []
    # print("S")
    for item in dataset_items:
        texts_enc.append(item["text_encode"])
        lens_texts.append(len(texts_enc[-1]))
        spectrograms.append(item["spectrogram"])
        # print(item["spectrogram"].shape)
        # print("SPEC", item["spectrogram"].shape)
        lens_specs.append(len(spectrograms[-1]))

        texts.append(item["text"])
    
    pad_specs = pad_sequence(spectrograms, batch_first=True)
    texts_enc = torch.cat(texts_enc)

    lens_specs = torch.tensor(lens_specs, dtype=torch.int32)
    lens_texts = torch.tensor(lens_texts, dtype=torch.int32)

    data = {
        "spectrograms":pad_specs,
        "lens_spectrograms": lens_specs,
        "texts_encode": texts_enc,
        "lens_texts": lens_texts,
        "text": texts
    }

    return data
    
    