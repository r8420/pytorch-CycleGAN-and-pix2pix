from .train_options import TrainOptions


class PaletteGANOptions(TrainOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.set_defaults(model='palettegan', dataset_mode='colorization_palette', input_nc=1, output_nc=2)
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
                            help='latent channels concatenated to L; 0 disables')
        parser.add_argument('--ca_window', type=int, default=9,
                            help='window size for Chromatic Attention local stats')
        parser.add_argument('--ca_down', type=int, default=4,
                            help='downsample factor for global attention')
        parser.add_argument('--mix_threshold', type=float, default=0.8,
                            help='probability threshold for using ground truth palette')
        parser.add_argument('--separate_te_opt', action='store_true',
                            help='use a separate optimizer for the palette encoder')
        parser.add_argument('--te_act', type=str, default='sigmoid',
                            choices=['sigmoid', 'softmax'],
                            help='activation for palette logits in TE; sigmoid will be renormalized to sum=1')
        parser.add_argument('--ref_image', type=str, default='', help='reference RGB image for palette at test time')
        parser.add_argument('--ref_dir', type=str, default='', help='directory of reference images matched by filename')
        return parser
