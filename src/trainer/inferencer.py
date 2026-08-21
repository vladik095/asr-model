import torch
from pyctcdecode import build_ctcdecoder
from torchmetrics.text import CharErrorRate, WordErrorRate
from tqdm import tqdm


class Inferencer:
    def __init__(
        self,
        model,
        config,
        device,
        dataloader,
        text_encoder,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.dataloader = dataloader
        self.text_encoder = text_encoder

        self.decoder = build_ctcdecoder(
            labels=self.text_encoder.vocab,
            kenlm_model_path=(
                "/kaggle/input/datasets/"
                "tuannguyenvananh/librispeech-4gram-language-model/"
                "4-gram-librispeech.bin"
            ),
        )

    def run_inference(self):
        self._from_pretraine()

        self.model.eval()

        p_bar = tqdm(
            self.dataloader,
            desc="Infer",
            total=len(self.dataloader),
        )

        wer_metric = WordErrorRate()
        cer_metric = CharErrorRate()

        with torch.no_grad():

            for batch in p_bar:

                spectrograms = batch["spectrograms"].to(self.device)
                spectrogram_length = batch[
                    "lens_spectrograms"
                ].to(self.device)

                output = self.model(
                    spectrograms,
                    spectrogram_length,
                )

                log_probs = output["log_probs"]
                log_probs_length = output["log_probs_length"]

                # (B, T, vocab)
                log_probs = log_probs.cpu()
                log_probs_length = log_probs_length.cpu()

                pred_texts = []

                for logits, length in zip(
                    log_probs,
                    log_probs_length,
                ):
                    logits = logits[:length]

                    text = self.decoder.decode(
                        logits.numpy(),
                        beam_width=100,
                        alpha=0.5,
                        beta=1.5,
                    )

                    pred_texts.append(text)

                wer_metric.update(
                    pred_texts,
                    batch["text"],
                )

                cer_metric.update(
                    pred_texts,
                    batch["text"],
                )

                print(
                    "WER_val:",
                    wer_metric.compute().item(),
                    "CER_val:",
                    cer_metric.compute().item(),
                )