from torch import Tensor, nn


class FeedForwardModule(nn.Module):
    """
    FeedForward module from:
    'Conformer: A Convolution-Augmented Transformer for Speech Recognition.'

    The first linear layer uses an expansion factor of 4,
    increasing the feature dimension by a factor of four.
    The second linear layer projects the expanded representation back to the model dimension.
    The feed-forward module uses the Swish activation function and a pre-norm residual connection.

    Args:
        d_model (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    """

    def __init__(
        self,
        d_model: int = 258,
        expansion_factor: int = 4,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, time_steps, d_model),
                where:
                    - batch_size: number of samples in the batch.
                    - time_steps: length of the input sequence.
                    - d_model: feature dimension of each time step.

        Returns:
            Tensor: Output tensor of shape (batch_size, time_steps, d_model).
        """
        return self.sequential(x)
