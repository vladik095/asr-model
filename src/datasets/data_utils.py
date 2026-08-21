from torch.utils.data import DataLoader

from src.datasets.base_dataset import BaseDataset
from src.datasets.collate import collate_fn


def get_dataloader(config, text_encoder, transform,data_path):
    
    dataset = BaseDataset(
        path_data_dir=data_path,
        transforms=transform,
        text_encoder=text_encoder,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    return dataloader