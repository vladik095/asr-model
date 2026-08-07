from torch import Tensor, nn

from .help_modules import GLU, Transpose


class DepthwiseConv1d(nn.Module):
    """
    Depthwise 1D convolution module.

    This module applies a separate convolutional filter to each input channel.
    It is implemented using Conv1d with groups equal to the number of input channels,
    which performs channel-wise convolution without mixing information between channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Amount of zero-padding added to both sides of the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:

        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
        )


    def forward(self, x: Tensor) -> Tensor:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, channels, time_steps).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, time_steps_out).
        """
        return self.conv(x)


class PointwiseConv1d(nn.Module):
    """
    Pointwise 1D convolution module.

    This module applies a 1D convolution with a kernel size of 1.
    Unlike regular convolutions, it does not change the temporal dimension
    and only performs a linear transformation across the channel dimension.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution.
        padding (int): Amount of zero-padding applied to the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:

        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, channels, time_steps).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, time_steps_out).
        """
        return self.conv(x)


class ConformerConvModule(nn.Module):
    """
    Convolution module from:
    'Conformer: A Convolution-Augmented Transformer for Speech Recognition.'

    This module captures local temporal dependencies using a depthwise
    convolution combined with pointwise convolutions and a gating mechanism.
    The module first expands the channel dimension using a pointwise convolution,
    applies a GLU activation, performs depthwise convolution over the time dimension,
    and projects the representation back to the original channel dimension.

    Args:
        in_channels (int): Number of input channels (d_model).
        kernel_size (int): Size of the depthwise convolution kernel.
        expansion_factor (int): Expansion factor for the pointwise convolution.
        dropout_p (float): Dropout probability.
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(
                in_channels,
                in_channels * expansion_factor,
                stride=1,
                padding=0,
            ),
            GLU(dim=1),
            DepthwiseConv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels).

        Returns:
            Tensor: Output tensor of shape (batch_size, time_steps, channels).
        """
        return self.sequential(inputs).transpose(1, 2)


class ConvSubsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Convolutional subsampling module for the Conformer encoder.

        This module reduces the temporal resolution of the input sequence using
        two strided 2D convolution layers. The convolutions extract local features
        from the input representation while reducing the time dimension. After the
        convolutional layers, the channel and frequency dimensions are flattened
        into a single feature dimension suitable for the Conformer encoder.

        Args:
            in_channels (int): Number of input channels for the convolution.
            out_channels (int): Number of output channels produced by the convolutional layers.
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inputs:
        x (Tensor): Input tensor of shape (batch_size, time_steps, feature_dim),
            where:
                - batch_size: Number of samples in the batch.
                - time_steps: Number of input frames.
                - feature_dim: Number of input acoustic features (e.g. mel bins).

        lengths (Tensor): Original sequence lengths of shape (batch_size,).
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, T, F = x.shape

        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(B, T, C * F)

        new_lengths = self.transform_input_lengths(lengths)

        return x, new_lengths

    def transform_input_lengths(self, input_lengths):
        """
        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """

        seq_len = input_lengths
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                seq_len = (
                    seq_len
                    + 2 * m.padding[0]
                    - m.dilation[0] * (m.kernel_size[0] - 1)
                    - 1
                ) // m.stride[0] + 1
        return seq_len.int()
