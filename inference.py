import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.datasets.data_utils import get_dataloader
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.trainer.inferencer import Inferencer
from src.transforms.spectrogram import LogMelSpectrogram


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def train(cfg: DictConfig) -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_encoder = CTCTextEncoder()
    transform = LogMelSpectrogram(
        sample_rate=cfg.spec_param.sr, n_mels=cfg.spec_param.n_mels
    )

    dataloader = get_dataloader(
        config=cfg,
        text_encoder=text_encoder,
        transform=transform,
        data_path=cfg.dataset.train_data_path,
    )

    model = instantiate(cfg.model)

    inferencer = Inferencer(
        model=model,
        config=cfg,
        device=device,
        dataloader=dataloader,
        text_encoder=text_encoder
    )

    inferencer.run_inference()

train()
