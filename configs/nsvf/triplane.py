_base_ = '../default.py'

expname = 'Lifestyle'
basedir = './logs/triplane_singlescene'

data = dict(
    datadir='./data/Synthetic_NSVF/Lifestyle',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=5000,
    lrate_k0=0,
    lrate_k0_xy=5e-4,
    lrate_k0_yz=5e-4,
    lrate_k0_zx=5e-4,
)

fine_model_and_render = dict(
    posbase_pe=0,
    use_coarse_geo=True,
    implicit_voxel_feat=False,
    feat_unfold=False, 
    rgbnet_dim=12,
)

fine_train = dict(
    lrate_k0=0,
    lrate_k0_xy=5e-4,
    lrate_k0_yz=5e-4,
    lrate_k0_zx=5e-4,
    # ray_sampler='random',
)



