# For a perceptually oriented variant (ours2), set: --lambda_adv 0.1
python train.py --model palettegan --dataset_mode colorization_palette \
    --dataroot DATA_ROOT --name palettegan_exp \
    --palette_bins 16 --palette_sigma 0.1 \
    --curriculum_iters 100000 \
    --ca_window 9 --ca_down 4 \
    --lambda_reg 5.0 --lambda_rec1 5.0 --lambda_rec2 1.0 \
    --lambda_rg 1.0 --lambda_adv 1.0
