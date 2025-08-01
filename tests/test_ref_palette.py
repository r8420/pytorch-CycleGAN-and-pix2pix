import torch
from argparse import Namespace
from models.palette_gan_model import PaletteGANModel
from models.networks.palette_modules import PaletteHistogram
from PIL import Image
import tempfile


def make_opt():
    return Namespace(
        dataroot='.', name='temp', checkpoints_dir='.', gpu_ids=[], isTrain=False,
        preprocess='none', load_size=64, crop_size=64, no_flip=True,
        palette_bins=4, palette_sigma=0.1,
        lambda_reg=1.0, lambda_rec1=1.0, lambda_rec2=1.0,
        lambda_rg=1.0, lambda_adv=1.0,
        curriculum_epochs=1, curriculum_iters=1,
        ngf=16, ndf=16,
        z_dim=0, ca_window=9, ca_down=4,
        mix_threshold=0.8, separate_te_opt=False,
        ref_image='', ref_dir='', te_act='sigmoid'
    )


def test_reference_palette():
    opt = make_opt()
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        img = Image.new('RGB', (32,32), (255,0,0))
        img.save(f.name)
        opt.ref_image = f.name
        model = PaletteGANModel(opt)
        model.eval()
        palette_dim = opt.palette_bins ** 2
        def fake_te(x):
            h_pred = torch.zeros(1, palette_dim)
            S = torch.zeros(1, opt.ngf * 2, x.shape[2] // 4, x.shape[3] // 4)
            return h_pred, S
        model.netTE = fake_te
        hist = PaletteHistogram(4,0.1)
        A = torch.zeros(1,1,32,32)
        B = torch.zeros(1,2,32,32)
        h = hist(B)
        inp = {'A': A, 'B': B, 'palette': h, 'A_paths': 'test.png'}
        model.set_input(inp)
        model.forward()
        assert torch.allclose(model.get_palette_in(), model.ref_h)

