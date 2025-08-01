import torch
from models.networks.palette_modules import PaletteEncoderTE

def run_act(act):
    te = PaletteEncoderTE(bins=4, palette_dim=16, act=act)
    L = torch.randn(2,1,32,32, requires_grad=True)
    h, S = te(L)
    assert h.shape == (2,16)
    assert torch.all(h >= 0)
    assert torch.all(h <= 1 + 1e-6)
    assert torch.allclose(h.sum(dim=1), torch.ones(2), atol=1e-5)
    loss = h.mean()
    loss.backward()
    grads = [p.grad for p in te.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_te_activation_sigmoid_softmax():
    for act in ['sigmoid', 'softmax']:
        run_act(act)
