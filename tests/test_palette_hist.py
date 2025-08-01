import torch
from models.networks.palette_modules import PaletteHistogram

def test_hist_grad_and_sum():
    hist = PaletteHistogram(bins=4, sigma=0.1)
    ab = torch.rand(2, 2, 8, 8, requires_grad=True)
    h = hist(ab)
    assert h.shape == (2, 16)
    h.sum().backward()
    assert ab.grad is not None
    assert torch.allclose(h.sum(dim=1), torch.ones(2), atol=1e-4)

def test_entropy_increase():
    hist = PaletteHistogram(bins=4, sigma=0.1)
    ab = torch.zeros(1, 2, 4, 4, requires_grad=True)
    h = hist(ab)
    ent = -(h * torch.log(h + 1e-8)).sum()
    assert ent.item() > 0

def test_hist_non_square():
    hist = PaletteHistogram(bins=4, sigma=0.1)
    ab = torch.randn(1, 2, 32, 48, requires_grad=True)
    h = hist(ab)
    assert h.shape == (1, 16)
    h.sum().backward()
    assert ab.grad is not None
