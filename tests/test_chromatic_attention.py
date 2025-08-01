import torch
from models.networks.palette_modules import ChromaticAttention

def test_chromatic_attention_shape_and_grad():
    torch.manual_seed(0)
    ca = ChromaticAttention(64)
    F = torch.randn(1, 64, 128, 128, requires_grad=True)
    S = torch.randn(1, 64, 128, 128, requires_grad=True)
    L = torch.randn(1, 1, 128, 128, requires_grad=True)
    out = ca(F, S, L)
    assert out.shape == F.shape
    out.mean().backward()
    assert F.grad is not None and S.grad is not None and L.grad is not None
