import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
from util.lab_color import rgb2lab
from models.networks.palette_modules import PaletteHistogram


class ColorizationPaletteDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        self.transform = get_transform(self.opt, convert=False)
        self.hist = PaletteHistogram(opt.palette_bins, opt.palette_sigma)

    def __getitem__(self, index):
        path = self.paths[index]
        im = Image.open(path).convert('RGB')
        im = self.transform(im)
        im_t = transforms.ToTensor()(im)  # RGB [0,1]
        lab_t = rgb2lab(im_t.unsqueeze(0)).squeeze(0)
        L = lab_t[[0], ...] / 50.0 - 1.0
        ab = lab_t[[1, 2], ...] / 110.0
        h = self.hist(ab.unsqueeze(0)).squeeze(0).detach()
        return {'A': L, 'B': ab, 'A_paths': path, 'B_paths': path, 'palette': h}

    def __len__(self):
        return len(self.paths)
