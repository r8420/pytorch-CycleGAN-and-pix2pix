import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PaletteProjectionDiscriminator(nn.Module):
    def __init__(self, input_nc=5, ndf=64, n_layers=3, palette_dim=256):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kw, 2, padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw)), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 1, padw)), nn.LeakyReLU(0.2, True)]
        self.features = nn.Sequential(*sequence)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.uncond = spectral_norm(nn.Linear(ndf * nf_mult, 1))
        self.proj = spectral_norm(nn.Linear(ndf * nf_mult, palette_dim))

    def forward(self, x: torch.Tensor, palette: torch.Tensor):
        feat = self.features(x)
        g = self.gap(feat).view(x.size(0), -1)
        score = self.uncond(g)
        proj = (self.proj(g) * palette).sum(dim=1, keepdim=True)
        return score + proj
