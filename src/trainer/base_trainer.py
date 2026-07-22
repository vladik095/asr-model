

import numpy as np
from tqdm import tqdm


class BaseTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 data_loader,
                 epochs,
                 loss,
                 writer,
                 device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.epochs = epochs
        self.loss = loss
        self.writer = writer
        self.device = device
    
    
    def train(self):
        for epoch in range(self.epochs):
            p_bar = tqdm(self.data_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
            all_params, trainable_params = self.cout_model_params(self.model)
            self.writer.log_parameters({
                "All params": all_params,
                "Trainable params": trainable_params,
            })
            for batch_idx, batch in enumerate(p_bar):
                
                spectrogram_length = batch["lens_spectrograms"].to(self.device)

                targets = batch["texts_encode"].to(self.device)
                target_lengths = batch["lens_texts"].to(self.device)

                
                output = self.model(batch)

                log_probs = output["log_probs"]


                loss = self.loss(
                    log_probs,
                    targets,
                    spectrogram_length,
                    target_lengths
                )

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # avg_loss.append(loss.item())
                p_bar.set_description(f"Epoch {epoch+1} | Loss: {loss:.4f}")
                
                # self.writer.log_metrics({
                #     "train_loss_batch": loss.item(),
                #     "learning_rate": self.optimizer.param_groups[0]['lr']
                # }, step=epoch * len(self.data_loader) + batch_idx)
            

            
    def cout_model_params(self, model):
        all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return all_params, trainable_params
    
    # def to_device(self, data_list)


        

