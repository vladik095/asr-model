import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items):
    
    spectrograms = []
    lens_specs = []

    texts_enc = []
    lens_texts = []

    texts = []
    
    for item in dataset_items:
        texts_enc.append(item["text_encode"])
        lens_texts.append(len(texts_enc[-1]))

        spectrograms.append(item["spectrogram"])
        # print("SPEC", item["spectrogram"].shape)
        lens_specs.append(len(spectrograms[-1]))

        texts.append(item["text"])
    
    pad_specs = pad_sequence(spectrograms,batch_first=True)
    texts_enc = torch.cat(texts_enc)

    data = {
        "spectrograms":pad_specs,
        "lens_spectrograms": lens_specs,
        "texts_encode": texts_enc,
        "lens_texts": lens_texts,
        "text": texts
    }

    return data
    
    