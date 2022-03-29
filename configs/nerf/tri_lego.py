_base_ = '../tri_default.py'

expname = 'rnd'
basedir = './logs/tri_dvgo/nerf_synthetic/lego'

data = dict(
    down=4,
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=40000,
    lrate_k0=0, 
    pg_scale=[],
    ray_sampler='random',
    # skip_zero_grad_fields=[],
)

fine_model_and_render = dict(
    implicit_voxel_feat=False,
    use_coarse_geo=False,
    # rgbnet_dim=12,
)