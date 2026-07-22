import os

import comet_ml
import hydra
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate

from src.datasets import BaseDataset, collate_fn
from src.models import BaselineModel, BaselineModelRNN
from src.text_encoder import CTCTextEncoder
from src.trainer import BaseTrainer
from src.transforms import LogMelSpectrogram


@hydra.main(version_base=None, config_path="src/configs", config_name="conf")
def train(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DATASET
    text_encoder = CTCTextEncoder()
    transform = LogMelSpectrogram(sample_rate=cfg.spec_param.sr, 
                                  n_mels=cfg.spec_param.n_mels)

    data = BaseDataset(path_data_dir=cfg.dataset_param.data_path, 
                       transforms=transform, 
                       text_encoder=text_encoder)
    
    print(cfg.spec_param.sr, cfg.spec_param.n_mels)
    # MODELS
    model = instantiate(cfg.model, n_tokens=len(text_encoder))
    print(model)
    # TRAIN PARAM
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # COMMET
    comet_ml.login()
    exp = comet_ml.start(project_name="my-awesome-project")
    exp.set_name("model2")

    data_loader = DataLoader(
        data, 
        batch_size=cfg.dataset_param.batch_size, 
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

train()
