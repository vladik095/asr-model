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

        self.best_val_wer = 1

    def train(self):
        self._train_process()

    def _train_process(self):
        parmas, _ = self.cout_model_params(self.model)
        print("ALL PARAMS: ", parmas)
        for epoch in range(self.start_epoch, self.epochs):
            print("EPOCH: ", epoch + 1)
            self._train_epoch(epoch)
            self._evaluation_epoch(epoch)
            if epoch != 0 and epoch % 5 == 0:
                name_checkpoint = f"checkpoint_{epoch+1}.pth"
                self.save_checkpoint(epoch, name_checkpoint)

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

            output = self.model(spectrograms, spectrogram_length)
            log_probs = output["log_probs"]
            log_probs_length = output["log_probs_length"]
            loss = self.loss(log_probs, targets, log_probs_length, target_lengths)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = log_probs.argmax(dim=-1)
            preds = preds.transpose(0, 1)
            pred_texts = self.get_pred_text(preds, log_probs_length)

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

                
                output = self.model(spectrograms, spectrogram_length)
                
                log_probs = output["log_probs"]
                log_probs_length = output["log_probs_length"]

                loss = self.loss(log_probs, targets, log_probs_length, target_lengths)

                preds = log_probs.argmax(dim=-1)
                preds = preds.transpose(0, 1)
                pred_texts = self.get_pred_text(preds, log_probs_length)

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

                print("log_probs:", log_probs.shape)
                print("log_probs_length:", log_probs_length)
                print("target_lengths:", target_lengths)

                print("unique argmax:",
                    torch.unique(log_probs.argmax(dim=-1)))

                print("argmax distribution:",
                    torch.bincount(
                        log_probs.argmax(dim=-1).flatten(),
                        minlength=log_probs.shape[-1]
                    ))

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
            wer = wer_metric.compute().item()
            if wer < self.best_val_wer:
                self.best_val_wer = wer
                name_checkpoint = "best.pth"
                self.save_checkpoint(cur_epoch, name_checkpoint)

    def cout_model_params(self, model):
        all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return all_params, trainable_params

    def get_pred_text(self, pred_tokens, spectrogram_length):
        decoded = []

        for pred, length in zip(pred_tokens, spectrogram_length):
            decoded.append(self.ctc_decoder.ctc_decode(pred[:length]))

        decoded_texts = [self.ctc_decoder.decode(tokens) for tokens in decoded]
        decoded_texts = ["".join(chars) for chars in decoded_texts]
        return decoded_texts

    def save_checkpoint(self, epoch, name_checkpoint):
        checkpoint_dir = Path(self.config.trainer.path_check)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_wer": self.best_val_wer
        }

        torch.save(checkpoint, checkpoint_dir / name_checkpoint)

    def load_checkpoint(self, path_to_checkpoint):
        
        checkpoint = torch.load(path_to_checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_wer = checkpoint["best_val_wer"]

    def resume_train(self, path_to_checkpoint):
        self.load_checkpoint(path_to_checkpoint)
        print("----------------------------------")
        print("The checkpoint loaded successfully")
        print(f"The model will continue training from epoch {self.start_epoch+1}.")
        print("----------------------------------")
        self._train_process()
