import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """PositionalEncoding module

    Maps v to positional encoding representation phi(v)

    Arguments:
        i_dim (int): input dimension for v
        N_freqs (int): #frequency to sample (default: 10)

    adapted from: https://github.com/yliess86/NeRF/blob/main/nerf/core/features.py
    and from: https://github.com/bebeal/mipnerf-pytorch/blob/main/model.py
    """

    def __init__(self, i_dim, a=0, b=4):
        super().__init__()
        self.i_dim = i_dim
        self.a = a
        self.b = b
        self.N_freqs = b - a

        self.o_dim = self.i_dim + (2 * self.N_freqs) * self.i_dim

        freq_bands = 2 ** torch.arange(a, b)
        self.register_buffer("freq_bands", freq_bands)
    
    def fourier_features(self, v):
        """Map v to positional encoding representation phi(v)

        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): fourier features (B, IFeatures + (2 * N_freqs) * IFeatures)
        """
        pe = [v]
        for freq in self.freq_bands:
            fv = freq * v
            pe += [torch.sin(fv), torch.cos(fv)]
        return torch.cat(pe, dim=-1)

    def forward(self, x):
        """Map x to positional encoding representation phi(v)
    
        Arguments:
            x (Tensor): input features (B, IFeatures)

        Returns:
            phi(x) (Tensor): fourier features (B, IFeatures + (2 * (b-a)) * IFeatures)
        """
        return self.fourier_features(x)