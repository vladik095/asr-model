from torch import nn


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), 
    which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
    



class Transpose(nn.Module):
    """ 
    Wrapper class of torch.transpose() for Sequential module. 
    """
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)