import os

import comet_ml
import hydra
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate

from src.datasets import BaseDataset, collate_fn
from src.models import BaselineModel, BaselineModelRNN
from src.text_encoder import CTCTextEncoder
from src.trainer import BaseTrainer
from src.transforms import LogMelSpectrogram


@hydra.main(version_base=None, config_path="src/configs", config_name="conf")
def my_app(cfg: DictConfig) -> None:

    # DATASET
    text_encoder = CTCTextEncoder()
    transform = LogMelSpectrogram()
    data = BaseDataset(path_data_dir=cfg.data, transforms=transform, text_encoder=text_encoder)

    # MODELS
    model = BaselineModel(n_feats=128, n_tokens=28, fc_hidden=128)
    # model = BaselineModelRNN(n_feats=128, n_tokens=28, hidden_size=256)

    # TRAIN PARAM
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # COMMET
    comet_ml.login()
    exp = comet_ml.start(project_name="my-awesome-project")
    exp.set_name("model2")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_loader = DataLoader(
        data, batch_size=12, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    trainer = BaseTrainer(
        model=model, 
        optimizer=optimizer, 
        data_loader=data_loader, 
        epochs=30, 
        loss=ctc_loss, 
        writer=exp,
        device=device
    )

    trainer.train()

my_app()
