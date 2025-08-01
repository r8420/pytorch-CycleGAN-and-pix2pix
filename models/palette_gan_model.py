import torch
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.palette_modules import (
    PaletteHistogram,
    PaletteEncoderTE,
    PaletteResnetGenerator,
)
from .networks.palette_discriminator import PaletteProjectionDiscriminator
from util.lab_color import lab2rgb, rgb2lab
from PIL import Image
import numpy as np
import os


class PaletteGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='colorization_palette', input_nc=1, output_nc=2)
        parser.add_argument('--palette_bins', type=int, default=16)
        parser.add_argument('--palette_sigma', type=float, default=0.1)
        parser.add_argument('--lambda_reg', type=float, default=5.0)
        parser.add_argument('--lambda_rec1', type=float, default=5.0)
        parser.add_argument('--lambda_rec2', type=float, default=1.0)
        parser.add_argument('--lambda_rg', type=float, default=1.0)
        parser.add_argument('--lambda_adv', type=float, default=1.0)
        parser.add_argument('--curriculum_epochs', type=int, default=40)
        parser.add_argument('--curriculum_iters', type=int, default=100000)
        parser.add_argument('--z_dim', type=int, default=0,
                            help='latent channels for generator input')
        parser.add_argument('--ca_window', type=int, default=9)
        parser.add_argument('--ca_down', type=int, default=4)
        parser.add_argument('--mix_threshold', type=float, default=0.8,
                            help='GT palette used if p_h > mix_threshold')
        parser.add_argument('--separate_te_opt', action='store_true',
                            help='use separate optimizer for TE')
        parser.add_argument('--ref_image', type=str, default='')
        parser.add_argument('--ref_dir', type=str, default='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['E_rec', 'G_reg', 'G_rec', 'G_adv', 'D_real', 'D_fake', 'Entropy']
        self.visual_names = [
            'real_A', 'fake_B_rgb', 'real_B_rgb', 'vis_palette_pred', 'vis_palette_gt', 'vis_palette_fake', 'vis_palette_ref'
        ]
        if self.isTrain:
            self.model_names = ['TE', 'G', 'D']
        else:
            self.model_names = ['TE', 'G']
        pal_dim = opt.palette_bins ** 2
        self.netTE = PaletteEncoderTE(opt.palette_bins, pal_dim, act=opt.te_act, s_out_ch=opt.ngf * 2).to(self.device)
        print(f'Palette TE activation: {opt.te_act}')
        in_nc = 1 + max(0, opt.z_dim)
        self.netG = PaletteResnetGenerator(in_nc, 2, opt.ngf, pal_dim,
                                           ca_down=opt.ca_down, z_dim=opt.z_dim,
                                           window_size=opt.ca_window).to(self.device)
        if self.isTrain:
            self.netD = PaletteProjectionDiscriminator(5, opt.ndf, palette_dim=pal_dim).to(self.device)
        self.hist = PaletteHistogram(opt.palette_bins, opt.palette_sigma)
        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            g_params = list(self.netG.parameters())
            if not opt.separate_te_opt:
                g_params += list(self.netTE.parameters())
            self.optimizer_G = torch.optim.Adam(g_params, lr=1e-4, betas=(0.0, 0.9))
            if opt.separate_te_opt:
                self.optimizer_TE = torch.optim.Adam(self.netTE.parameters(), lr=1e-4, betas=(0.0, 0.9))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=4e-4, betas=(0.0, 0.9))
            self.optimizers.append(self.optimizer_G)
            if opt.separate_te_opt:
                self.optimizers.append(self.optimizer_TE)
            self.optimizers.append(self.optimizer_D)
        self.global_step = 0
        if not self.isTrain and opt.ref_image:
            self.ref_h = self._load_palette_from_path(opt.ref_image)
        else:
            self.ref_h = None
        if not self.isTrain:
            self.ref_dir = opt.ref_dir
        else:
            self.ref_dir = ''

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.h = input['palette'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.h_pred, self.S = self.netTE(self.real_A)
        tau = max(0.0, 1.0 - float(self.global_step) / max(1, self.opt.curriculum_iters))
        self.tau = tau
        use_gt = False
        ref_h = None
        if not self.isTrain:
            if self.ref_h is not None:
                ref_h = self.ref_h
            elif self.ref_dir:
                base = os.path.basename(self.image_paths)
                cand = os.path.join(self.ref_dir, base)
                if os.path.exists(cand):
                    ref_h = self._load_palette_from_path(cand)
                else:
                    files = [f for f in os.listdir(self.ref_dir) if f.lower().endswith(('jpg','png','jpeg'))]
                    if files:
                        ref_h = self._load_palette_from_path(os.path.join(self.ref_dir, files[0]))
        if ref_h is not None:
            palette_in = ref_h
            self.vis_palette_ref = ref_h.view(-1, 1, self.opt.palette_bins, self.opt.palette_bins)
        else:
            p_h = tau + (1.0 - tau) * torch.rand(1, device=self.device)
            use_gt = p_h.item() > self.opt.mix_threshold
            palette_in = self.h if use_gt else self.h_pred
            self.vis_palette_ref = torch.zeros_like(self.h_pred.view(-1, 1, self.opt.palette_bins, self.opt.palette_bins))
        self.palette_in = palette_in
        # Build detached inputs for G when using separate_te_opt (unless using a reference palette)
        if self.opt.separate_te_opt and (ref_h is None):
            self.palette_in_g = self.palette_in.detach()
            self.S_g = self.S.detach()
        else:
            self.palette_in_g = self.palette_in
            self.S_g = self.S
        self.use_gt_flag = use_gt
        z = None
        if self.opt.z_dim > 0:
            B, _, H, W = self.real_A.shape
            z = torch.randn(B, self.opt.z_dim, H, W, device=self.device)
        self.fake_B = self.netG(self.real_A, self.palette_in_g, self.S_g, z=z)
        lab_fake = torch.cat([(self.real_A + 1.0) * 50.0, self.fake_B * 110.0], 1)
        self.fake_B_rgb = lab2rgb(lab_fake)
        lab_real = torch.cat([(self.real_A + 1.0) * 50.0, self.real_B * 110.0], 1)
        self.real_B_rgb = lab2rgb(lab_real)
        self.vis_palette_pred = self.h_pred.view(-1, 1, self.opt.palette_bins, self.opt.palette_bins)
        if hasattr(self, 'hist'):
            self.vis_palette_fake = self.hist(self.fake_B).view(-1, 1, self.opt.palette_bins, self.opt.palette_bins)
        else:
            self.vis_palette_fake = torch.zeros_like(self.vis_palette_pred)
        self.vis_palette_gt = self.h.view(-1, 1, self.opt.palette_bins, self.opt.palette_bins)

    def backward_D(self):
        fake_in = torch.cat([self.fake_B.detach(), self.fake_B_rgb.detach()], dim=1)
        real_in = torch.cat([self.real_B.detach(), self.real_B_rgb.detach()], dim=1)
        pred_fake = self.netD(fake_in, self.palette_in.detach())
        pred_real = self.netD(real_in, self.h)
        self.loss_D_fake = F.relu(1 + pred_fake).mean()
        self.loss_D_real = F.relu(1 - pred_real).mean()
        loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        loss_D.backward()

    def backward_G(self):
        fake_in = torch.cat([self.fake_B, self.fake_B_rgb], dim=1)
        pred_fake = self.netD(fake_in, self.palette_in_g)
        self.loss_G_adv = -pred_fake.mean() * self.opt.lambda_adv
        self.loss_G_reg = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_reg
        h_fake = self.hist(self.fake_B)
        self.loss_G_rec = self.criterionL1(h_fake, self.h) * self.opt.lambda_rec2
        entropy = -(self.h_pred * torch.log(self.h_pred + 1e-8)).sum(dim=1).mean()
        self.loss_Entropy = -entropy * self.opt.lambda_rg
        self.loss_E_rec = self.criterionL1(self.h_pred, self.h) * self.opt.lambda_rec1
        loss_G = self.loss_G_adv + self.loss_G_reg + self.loss_G_rec + self.loss_Entropy + self.loss_E_rec
        if not self.opt.separate_te_opt:
            loss_G.backward()
        return loss_G

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        if self.opt.separate_te_opt:
            self.backward_G()  # compute losses

            # TE-only step
            self.optimizer_TE.zero_grad()
            (self.loss_E_rec + self.loss_Entropy).backward(retain_graph=True)
            self.optimizer_TE.step()

            # G-only step; ensure no residual grads on TE
            for p in self.netTE.parameters():
                p.grad = None
            self.optimizer_G.zero_grad()
            (self.loss_G_adv + self.loss_G_reg + self.loss_G_rec).backward()
            self.optimizer_G.step()
        else:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        self.global_step += 1
        if self.global_step % self.opt.print_freq == 0:
            ent = -self.loss_Entropy.item() / max(self.opt.lambda_rg, 1e-8)
            print(f'tau={self.tau:.3f} use_gt={self.use_gt_flag} entropy={ent:.3f}')

    def get_tau(self):
        return max(0.0, 1.0 - float(self.global_step) / max(1, self.opt.curriculum_iters))

    def get_palette_in(self):
        return getattr(self, 'palette_in', None)

    def _load_palette_from_path(self, path: str):
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            return None
        np_im = np.array(img).astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(np_im.transpose(2,0,1)).unsqueeze(0)
        try:
            lab_t = rgb2lab(rgb_t)
        except Exception:
            r, g, b = rgb_t[:,0], rgb_t[:,1], rgb_t[:,2]
            rgb_lin = torch.where(rgb_t <= 0.04045, rgb_t/12.92, ((rgb_t+0.055)/1.055)**2.4)
            r, g, b = rgb_lin[:,0], rgb_lin[:,1], rgb_lin[:,2]
            X = r*0.412453 + g*0.357580 + b*0.180423
            Y = r*0.212671 + g*0.715160 + b*0.072169
            Z = r*0.019334 + g*0.119193 + b*0.950227
            X = X / 0.95047
            Z = Z / 1.08883
            epsilon = 0.008856
            kappa = 903.3
            fx = torch.where(X>epsilon, X.pow(1/3), (kappa*X+16)/116)
            fy = torch.where(Y>epsilon, Y.pow(1/3), (kappa*Y+16)/116)
            fz = torch.where(Z>epsilon, Z.pow(1/3), (kappa*Z+16)/116)
            L = 116*fy -16
            a_ = 500*(fx - fy)
            b_ = 200*(fy - fz)
            lab_t = torch.stack([L,a_,b_],1)
        ab = lab_t[0,1:3] / 110.0
        with torch.no_grad():
            h = self.hist(ab.unsqueeze(0)).squeeze(0)
        return h.to(self.device)
