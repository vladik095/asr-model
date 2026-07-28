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

from src.datasets import BaseDataset, collate_fn
from src.models import DeepSpeech, DeepSpeech2
from src.text_encoder import CTCTextEncoder
from src.trainer import BaseTrainer
from src.transforms import LogMelSpectrogram


@hydra.main(version_base=None, config_path="src/configs", config_name="conf")
def train(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_encoder = CTCTextEncoder()
    transform = LogMelSpectrogram(
        sample_rate=cfg.spec_param.sr, n_mels=cfg.spec_param.n_mels
    )

    # DATASETS
    train_dataset = BaseDataset(
        path_data_dir=cfg.dataset.train_data_path,
        transforms=transform,
        text_encoder=text_encoder,
    )
    val_dataset = BaseDataset(
        path_data_dir=cfg.dataset.val_data_path,
        transforms=transform,
        text_encoder=text_encoder,
    )
    
    # DATALOADERS
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # TRAIN PARAM
    # model = instantiate(cfg.model, n_tokens=len(text_encoder))
    model = DeepSpeech2()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # COMMET
    comet_ml.login()
    exp = comet_ml.start(project_name="my-awesome-project")
    exp.set_name("deep speech")

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        train_data_loader=train_dataloader,
        val_data_loader=val_dataloader,
        config=cfg,
        loss=ctc_loss,
        writer=exp,
        device=device,
        ctc_decode=text_encoder,
    )

    if cfg.trainer.load_checkpoint:
        trainer.resume_train(cfg.trainer.resume_from)
    else:
        trainer.train()


train()
