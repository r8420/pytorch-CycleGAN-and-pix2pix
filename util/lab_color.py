import torch
import torch.nn.functional as F
import numpy as np

try:
    import kornia.color as kcolor
except Exception:  # pragma: no cover
    kcolor = None

try:
    from skimage import color as skcolor
except Exception:  # pragma: no cover
    skcolor = None


def lab2rgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert Lab tensor (L[0,100], ab[-128,127]) to RGB [0,1]."""
    if kcolor is not None:
        return kcolor.lab_to_rgb(lab)
    if skcolor is None:
        raise RuntimeError('No Lab converter available')
    lab_np = lab.permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
    rgb_np = np.stack([skcolor.lab2rgb(l) for l in lab_np], axis=0)
    rgb = torch.from_numpy(rgb_np).permute(0, 3, 1, 2)
    return rgb


def rgb2lab(rgb: torch.Tensor) -> torch.Tensor:
    if kcolor is not None:
        return kcolor.rgb_to_lab(rgb)
    if skcolor is None:
        raise RuntimeError('No RGB2Lab converter available')
    rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
    lab_np = np.stack([skcolor.rgb2lab(r) for r in rgb_np], axis=0)
    lab = torch.from_numpy(lab_np).permute(0, 3, 1, 2)
    return lab


def palette_to_map(palette: torch.Tensor, bins: int = 16, size: int = 128) -> torch.Tensor:
    """Convert flat palette distribution to heatmap for visualization."""
    heat = palette.view(-1, 1, bins, bins)
    if size:
        heat = F.interpolate(heat, size=(size, size), mode='bilinear', align_corners=False)
    return heat
