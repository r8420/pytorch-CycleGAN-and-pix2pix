import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional


class PaletteHistogram(nn.Module):
    def __init__(self, bins=16, sigma=0.1):
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        centers = torch.linspace(-1.0, 1.0, bins)
        self.register_buffer('a_centers', centers.view(1, bins, 1, 1))
        self.register_buffer('b_centers', centers.view(1, 1, bins, 1))

    def forward(self, ab: torch.Tensor) -> torch.Tensor:
        """Compute differentiable 2D histogram using inverse-quadratic kernel."""
        B, _, H, W = ab.shape
        a = ab[:, 0:1].unsqueeze(1)  # (B,1,1,H,W)
        b = ab[:, 1:2].unsqueeze(1)  # (B,1,1,H,W)
        a_centers = self.a_centers.view(1, self.bins, 1, 1, 1)
        b_centers = self.b_centers.view(1, 1, self.bins, 1, 1)
        ka = 1.0 / (1.0 + ((a - a_centers) / self.sigma) ** 2)
        kb = 1.0 / (1.0 + ((b - b_centers) / self.sigma) ** 2)
        weights = ka * kb  # (B,Na,Nb,H,W)
        hist = weights.sum(dim=(3, 4))  # (B,Na,Nb)
        hist = hist.reshape(B, -1)
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        return hist


class PaletteNorm(nn.Module):
    def __init__(self, num_features: int, palette_dim: int = 256):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = spectral_norm(nn.Linear(palette_dim, num_features))
        self.beta = spectral_norm(nn.Linear(palette_dim, num_features))

    def forward(self, x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        g = self.gamma(palette).view(x.size(0), -1, 1, 1)
        b = self.beta(palette).view(x.size(0), -1, 1, 1)
        return out * g + b


class ChromaticAttention(nn.Module):
    def __init__(self, channels: int, window_size: int = 9, attn_down: int = 4):
        super().__init__()
        inter = max(8, channels // 8)
        self.q = spectral_norm(nn.Conv2d(channels, inter, 1))
        self.k = spectral_norm(nn.Conv2d(channels, inter, 1))
        self.v = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.psi = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(channels, channels, 1)),
        )
        self.fuse = spectral_norm(nn.Conv2d(channels * 2, channels, 1))
        self.win = window_size
        self.attn_down = attn_down
        self._warned = False

    def forward(self, Fmap: torch.Tensor, S: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        B, C, H, W = Fmap.shape
        down = self.attn_down
        Sd = F.avg_pool2d(S, down, down)
        Fd = F.avg_pool2d(Fmap, down, down)
        _, _, Hs, Ws = Sd.shape
        while Hs * Ws > 4096:
            down *= 2
            Sd = F.avg_pool2d(S, down, down)
            Fd = F.avg_pool2d(Fmap, down, down)
            _, _, Hs, Ws = Sd.shape
            if not self._warned:
                print(f'ChromaticAttention: increasing attn_down to {down} for memory safety')
                self._warned = True
        q = F.normalize(self.q(Sd).view(B, -1, Hs * Ws), dim=1)
        k = F.normalize(self.k(Sd).view(B, -1, Hs * Ws), dim=1)
        v = self.v(Fd).view(B, C, Hs * Ws)
        att = torch.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)
        Fg_small = torch.bmm(v, att).view(B, C, Hs, Ws)
        Fg = F.interpolate(Fg_small, size=(H, W), mode='bilinear', align_corners=False)

        box = self.win
        mu_F = F.avg_pool2d(Fmap, box, 1, box // 2)
        mu_L = F.avg_pool2d(L, box, 1, box // 2)
        cov_FL = F.avg_pool2d(Fmap * L, box, 1, box // 2) - mu_F * mu_L
        var_L = F.avg_pool2d(L * L, box, 1, box // 2) - mu_L * mu_L
        A = cov_FL / (var_L + 1e-4)
        A = self.psi(A)
        L128 = F.interpolate(L, size=(H, W), mode='bilinear', align_corners=False)
        Bmap = mu_F - A * mu_L
        Fl = A * L128 + Bmap

        out = Fmap + self.fuse(torch.cat([Fg, Fl], 1))
        return out


class ResnetBlockPN(nn.Module):
    def __init__(self, dim: int, palette_dim: int):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(dim, dim, 3, padding=1))
        self.pn1 = PaletteNorm(dim, palette_dim)
        self.conv2 = spectral_norm(nn.Conv2d(dim, dim, 3, padding=1))
        self.pn2 = PaletteNorm(dim, palette_dim)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.pn1(self.conv1(x), p))
        y = self.pn2(self.conv2(y), p)
        return x + y


class PaletteEncoderTE(nn.Module):
    def __init__(self, bins: int = 16, palette_dim: int = 256, act: str = 'sigmoid', s_out_ch: Optional[int] = None):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 4, 2, 1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.ReLU(True),
        )
        self.s_proj = spectral_norm(nn.Conv2d(512, s_out_ch, 1)) if s_out_ch is not None else None
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(512, 512)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(512, palette_dim)),
        )

    def forward(self, L: torch.Tensor):
        feat = self.conv(L)
        S_base = feat
        if self.s_proj is not None:
            S_base = self.s_proj(S_base)
        S = F.interpolate(S_base, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.gap(feat).view(feat.size(0), -1)
        logits = self.mlp(x)
        if self.act == 'softmax':
            h = torch.softmax(logits, dim=1)
        else:
            h = torch.sigmoid(logits)
            h = h / (h.sum(dim=1, keepdim=True) + 1e-8)
        return h, S


class PaletteResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=2, ngf=64, palette_dim=256, n_blocks=9, ca_down=4, z_dim=0, window_size=9):
        super().__init__()
        self.z_dim = z_dim
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = spectral_norm(nn.Conv2d(input_nc, ngf, 7))
        self.pn1 = PaletteNorm(ngf, palette_dim)
        self.down1 = spectral_norm(nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1))
        self.pn2 = PaletteNorm(ngf * 2, palette_dim)
        self.down2 = spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1))
        self.pn3 = PaletteNorm(ngf * 4, palette_dim)
        self.resblocks = nn.ModuleList([ResnetBlockPN(ngf * 4, palette_dim) for _ in range(n_blocks)])
        self.up1 = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1))
        self.pn4 = PaletteNorm(ngf * 2, palette_dim)
        self.ca = ChromaticAttention(ngf * 2, window_size=window_size, attn_down=ca_down)
        self.up2 = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1))
        self.pn5 = PaletteNorm(ngf, palette_dim)
        self.pad_out = nn.ReflectionPad2d(3)
        self.conv_out = spectral_norm(nn.Conv2d(ngf, output_nc, 7))

    def forward(self, L: torch.Tensor, palette: torch.Tensor, S: torch.Tensor, z: torch.Tensor = None):
        x_in = L if z is None else torch.cat([L, z], dim=1)
        x = self.pad1(x_in)
        x = F.relu(self.pn1(self.conv1(x), palette))
        x = F.relu(self.pn2(self.down1(x), palette))
        x = F.relu(self.pn3(self.down2(x), palette))
        for rb in self.resblocks:
            x = rb(x, palette)
        x = F.relu(self.pn4(self.up1(x), palette))
        L128 = F.interpolate(L, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.ca(x, F.interpolate(S, size=x.shape[2:], mode='bilinear', align_corners=False), L128)
        x = F.relu(self.pn5(self.up2(x), palette))
        x = self.conv_out(self.pad_out(x))
        return torch.tanh(x)
