import torch
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

    def run_inference(self):
        self._from_pretraine()

        self.model.eval()
        p_bar = tqdm(self.dataloader, desc="Infer", total=len(self.dataloader))
        with torch.no_grad():

            wer_metric = WordErrorRate()
            cer_metric = CharErrorRate()

            for _, batch in enumerate(p_bar):

                spectrograms = batch["spectrograms"].to(self.device)
                spectrogram_length = batch["lens_spectrograms"].to(self.device)
                output = self.model(spectrograms, spectrogram_length)

                log_probs = output["log_probs"]
                log_probs_length = output["log_probs_length"]

                preds = log_probs.argmax(dim=-1)
                preds = preds.transpose(0, 1)
                pred_texts = self.get_pred_text(
                    self.text_encoder, preds, log_probs_length
                )

                wer_metric.update(pred_texts, batch["text"])
                cer_metric.update(pred_texts, batch["text"])

                print(
                    "WER_val ",
                    wer_metric.compute().item(),
                    "CER_val ",
                    cer_metric.compute().item(),
                )

    def _from_pretraine(self):

        checkpoint = torch.load(
            self.config.inferencer.from_pretrained, map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def get_pred_text(self, ctc_decoder, pred_tokens, spectrogram_length):
        decoded = []

        for pred, length in zip(pred_tokens, spectrogram_length):
            decoded.append(ctc_decoder.ctc_decode(pred[:length]))

        decoded_texts = [ctc_decoder.decode(tokens) for tokens in decoded]
        decoded_texts = ["".join(chars) for chars in decoded_texts]
        return decoded_texts
