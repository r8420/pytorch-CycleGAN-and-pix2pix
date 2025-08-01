import torch
from argparse import Namespace
from models.palette_gan_model import PaletteGANModel
from models.networks.palette_modules import PaletteHistogram


def make_opt():
    return Namespace(
        dataroot='.', name='temp', checkpoints_dir='.', gpu_ids=[], isTrain=True,
        preprocess='none', load_size=64, crop_size=64, no_flip=True,
        palette_bins=4, palette_sigma=0.1,
        lambda_reg=1.0, lambda_rec1=1.0, lambda_rec2=1.0,
        lambda_rg=1.0, lambda_adv=1.0,
        curriculum_epochs=1, curriculum_iters=10,
        ngf=16, ndf=16,
        z_dim=0, ca_window=9, ca_down=4,
        mix_threshold=0.8, separate_te_opt=False, ref_image='', ref_dir='',
        print_freq=1, te_act='sigmoid'
    )


def test_curriculum_tau_decreases():
    opt = make_opt()
    model = PaletteGANModel(opt)
    # Replace TE to avoid heavy convs and ensure CA input shape matches
    palette_dim = opt.palette_bins ** 2
    def fake_te(x):
        h = torch.full((1, palette_dim), 1.0 / palette_dim)
        S = torch.zeros(1, opt.ngf * 2, x.shape[2] // 4, x.shape[3] // 4)
        return h, S
    model.netTE = fake_te
    # Avoid numpy dependency in lab_color
    import models.palette_gan_model as pm
    pm.lab2rgb = lambda x: torch.zeros(x.size(0), 3, x.size(2), x.size(3))
    hist = PaletteHistogram(4, 0.1)
    A = torch.zeros(1, 1, 32, 32)
    B = torch.zeros(1, 2, 32, 32)
    h = hist(B)
    inp = {'A': A, 'B': B, 'palette': h, 'A_paths': 'x'}
    prev_tau = model.get_tau()
    for i in range(5):
        model.set_input(inp)
        model.optimize_parameters()
        assert model.global_step == i + 1
        tau = model.get_tau()
        assert tau <= prev_tau + 1e-6
        prev_tau = tau

def test_palette_in_matches_source():
    opt = make_opt()
    model = PaletteGANModel(opt)
    palette_dim = opt.palette_bins ** 2
    def fake_te(x):
        h_pred = torch.full((1, palette_dim), 1.0 / palette_dim * 0.5)
        S = torch.zeros(1, opt.ngf * 2, x.shape[2] // 4, x.shape[3] // 4)
        return h_pred, S
    model.netTE = fake_te
    import models.palette_gan_model as pm
    pm.lab2rgb = lambda x: torch.zeros(x.size(0), 3, x.size(2), x.size(3))
    hist = PaletteHistogram(4, 0.1)
    A = torch.zeros(1, 1, 32, 32)
    B = torch.zeros(1, 2, 32, 32)
    h = hist(B)
    inp = {'A': A, 'B': B, 'palette': h, 'A_paths': 'x'}
    model.global_step = 0
    with torch.no_grad():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model.set_input(inp)
            model.forward()
            assert torch.allclose(model.get_palette_in(), h)
    model.global_step = opt.curriculum_iters + 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        model.set_input(inp)
        model.forward()
        assert torch.allclose(model.get_palette_in(), model.h_pred)

def test_palette_selection_probability():
    opt = make_opt()
    model = PaletteGANModel(opt)
    palette_dim = opt.palette_bins ** 2
    def fake_te(x):
        h = torch.full((1, palette_dim), 1.0 / palette_dim)
        S = torch.zeros(1, opt.ngf * 2, x.shape[2] // 4, x.shape[3] // 4)
        return h, S
    model.netTE = fake_te
    import models.palette_gan_model as pm
    pm.lab2rgb = lambda x: torch.zeros(x.size(0), 3, x.size(2), x.size(3))
    hist = PaletteHistogram(4, 0.1)
    A = torch.zeros(1, 1, 32, 32)
    B = torch.zeros(1, 2, 32, 32)
    h = hist(B)
    inp = {'A': A, 'B': B, 'palette': h, 'A_paths': 'x'}
    model.global_step = 0
    cnt = 0
    for i in range(20):
        torch.manual_seed(i)
        model.set_input(inp)
        model.forward()
        if torch.allclose(model.get_palette_in(), h):
            cnt += 1
    assert cnt == 20
    model.global_step = opt.curriculum_iters
    cnt = 0
    for i in range(100):
        torch.manual_seed(i)
        model.set_input(inp)
        model.forward()
        if torch.allclose(model.get_palette_in(), h):
            cnt += 1
    assert abs(cnt - int((1 - opt.mix_threshold) * 100)) <= 30
