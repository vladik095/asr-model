from pathlib import Path

import numpy as np
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate
from tqdm import tqdm


class BaseTrainer:
    def __init__(
        self,
        model,
        optimizer,
        train_data_loader,
        val_data_loader,
        config,
        loss,
        writer,
        device,
        ctc_decode,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.config = config
        self.epochs = config.trainer.epochs
        self.loss = loss
        self.writer = writer
        self.device = device
        self.ctc_decoder = ctc_decode

        self.start_epoch = 0

    def train(self):
        self._train_process()

    def _train_process(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("EPOCH: ", epoch + 1)
            self._train_epoch(epoch)
            self._evaluation_epoch(epoch)
            self.save_checkpoint(epoch)
            break
            print()

    def _train_epoch(self, cur_epoch):
        self.model.train()
        p_bar = tqdm(
            self.train_data_loader, desc="Train", total=len(self.train_data_loader)
        )

        wer_metric = WordErrorRate()
        cer_metric = CharErrorRate()

        for batch_idx, batch in enumerate(p_bar):
            spectrograms = batch["spectrograms"].to(self.device)
            spectrogram_length = batch["lens_spectrograms"].to(self.device)

            targets = batch["texts_encode"].to(self.device)
            target_lengths = batch["lens_texts"].to(self.device)

            output = self.model(spectrograms)
            log_probs = output["log_probs"]

            loss = self.loss(log_probs, targets, spectrogram_length, target_lengths)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = log_probs.argmax(dim=-1)
            preds = preds.transpose(0, 1)
            pred_texts = self.get_pred_text(preds, spectrogram_length)

            wer_metric.update(pred_texts, batch["text"])
            cer_metric.update(pred_texts, batch["text"])

            self.writer.log_metrics(
                {
                    "loss_train": loss,
                    "WER_train": wer_metric.compute().item(),
                    "CER_train": cer_metric.compute().item(),
                },
                step=len(self.train_data_loader) * cur_epoch + batch_idx,
            )

    def _evaluation_epoch(self, cur_epoch):
        self.model.eval()
        p_bar = tqdm(self.val_data_loader, desc="Val", total=len(self.val_data_loader))

        with torch.no_grad():

            wer_metric = WordErrorRate()
            cer_metric = CharErrorRate()

            for batch_idx, batch in enumerate(p_bar):
                spectrograms = batch["spectrograms"].to(self.device)
                spectrogram_length = batch["lens_spectrograms"].to(self.device)

                targets = batch["texts_encode"].to(self.device)
                target_lengths = batch["lens_texts"].to(self.device)

                output = self.model(spectrograms)
                log_probs = output["log_probs"]

                loss = self.loss(log_probs, targets, spectrogram_length, target_lengths)

                preds = log_probs.argmax(dim=-1)
                preds = preds.transpose(0, 1)
                pred_texts = self.get_pred_text(preds, spectrogram_length)

                wer_metric.update(pred_texts, batch["text"])
                cer_metric.update(pred_texts, batch["text"])

                self.writer.log_metrics(
                    {
                        "loss_val": loss,
                        "WER_val": wer_metric.compute().item(),
                        "CER_val": cer_metric.compute().item(),
                    },
                    step=len(self.val_data_loader) * cur_epoch + batch_idx,
                )

                if batch_idx < 3:
                    examples = []
                    for target, pred in zip(batch["text"], pred_texts):
                        examples.append(
                            [
                                cur_epoch,
                                target,
                                pred,
                                cer_metric.compute().item(),
                                wer_metric.compute().item(),
                            ]
                        )

                    self.writer.log_table(
                        filename="predictions.csv",
                        tabular_data=examples,
                        headers=[
                            "epoch",
                            "target",
                            "prediction",
                            "cer",
                            "wer",
                        ],
                    )

    # def cout_model_params(self, model):
    #     all_params = sum(p.numel() for p in model.parameters())
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #     return all_params, trainable_params

    def get_pred_text(self, pred_tokens, spectrogram_length):
        decoded = []

        for pred, length in zip(pred_tokens, spectrogram_length):
            decoded.append(self.ctc_decoder.ctc_decode(pred[:length]))

        decoded_texts = [self.ctc_decoder.decode(tokens) for tokens in decoded]
        decoded_texts = ["".join(chars) for chars in decoded_texts]
        return decoded_texts

    def save_checkpoint(self, epoch):
        checkpoint_dir = Path(self.config.trainer.path_check)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, checkpoint_dir / f"checkpoint_{epoch+1}.pth")

    def load_checkpoint(self, filename):
        checkpoint_path = Path(self.config.trainer.path_check) / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1

    def resume_train(self, path_model):
        self.load_checkpoint(path_model)
        print("----------------------------------")
        print("The checkpoint loaded successfully")
        print(f"The model will continue training from epoch {self.start_epoch+1}.")
        print("----------------------------------")
        self._train_process()
