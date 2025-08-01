import torch
from argparse import Namespace
from models.palette_gan_model import PaletteGANModel
from models.networks.palette_modules import PaletteHistogram
import models.palette_gan_model as pm


def make_opt(sep):
    return Namespace(
        dataroot='.', name='temp', checkpoints_dir='.', gpu_ids=[], isTrain=True,
        preprocess='none', load_size=32, crop_size=32, no_flip=True,
        palette_bins=4, palette_sigma=0.1,
        lambda_reg=1.0, lambda_rec1=1.0, lambda_rec2=1.0,
        lambda_rg=1.0, lambda_adv=1.0,
        curriculum_epochs=1, curriculum_iters=1,
        ngf=16, ndf=16,
        z_dim=0, ca_window=9, ca_down=4,
        mix_threshold=1.1,
        separate_te_opt=sep, ref_image='', ref_dir='',
        print_freq=1, te_act='sigmoid'
    )


def run_case(sep):
    opt = make_opt(sep)
    model = PaletteGANModel(opt)
    pm.lab2rgb = lambda x: torch.zeros(x.size(0), 3, x.size(2), x.size(3))
    hist = PaletteHistogram(opt.palette_bins, opt.palette_sigma)
    A = torch.zeros(1, 1, 32, 32)
    B = torch.zeros(1, 2, 32, 32)
    h = hist(B)
    model.set_input({'A': A, 'B': B, 'palette': h, 'A_paths': 'x'})
    model.forward()
    return model.palette_in.requires_grad, model.palette_in_g.requires_grad


def test_palette_detach_behavior():
    req_p, req_pg = run_case(sep=False)
    assert req_p is True and req_pg is True

    req_p, req_pg = run_case(sep=True)
    assert req_p is True and req_pg is False

