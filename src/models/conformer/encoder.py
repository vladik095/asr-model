import torch
from torch import nn

from .attention import MultiHeadedSelfAttentionModule
from .convolution import ConformerConvModule, ConvSubsampling
from .feed_forward import FeedForwardModule


class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        depthwise_conv_kernel_size: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model=d_model, dropout_p=dropout_p)
        self.attention = MultiHeadedSelfAttentionModule(
            d_model=d_model, num_heads=num_heads, dropout=dropout_p
        )
        self.conv_module = ConformerConvModule(
            in_channels=d_model,
            kernel_size=depthwise_conv_kernel_size,
            dropout_p=dropout_p,
        )
        self.ffn2 = FeedForwardModule(d_model=d_model, dropout_p=dropout_p)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        input_pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        residual = inputs
        x = self.ffn1(inputs)
        x = x * 0.5 + residual

        residual = x
        x = self.attention(x, input_pad_mask)
        x = x + residual

        residual = x
        x = self.conv_module(x)
        x = x + residual

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        d_model: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.conv_subsample = ConvSubsampling(in_channels=1, out_channels=d_model)
        self.input_projection = nn.Sequential(
            nn.Linear(d_model * (((input_dim - 1) // 2 - 1) // 2), d_model),
            nn.Dropout(p=dropout_p),
        )

        self.conformer_layers = nn.ModuleList(
            [
                ConformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        input_pad_mask = self._build_pad_mask(output_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.conformer_layers:
            outputs = layer(outputs, input_pad_mask)

        return outputs, output_lengths

    @staticmethod
    def _build_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_length = int(torch.max(lengths).item())
        padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
            batch_size, max_length
        ) >= lengths.unsqueeze(1)

        return padding_mask


class Conformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        d_model: int,
        num_layers: int,
        num_classes: int,
        depthwise_conv_kernel_size: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout_p=dropout_p,
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        log_probs = nn.functional.log_softmax(outputs, dim=-1)

        return {
            "log_probs": log_probs.transpose(0, 1),
            "log_probs_length": encoder_output_lengths,
        }
    