_base_ = '../default.py'

expname = 'debug'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/hydeng/data/NeRF_data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=0,
)

fine_model_and_render = dict(
    posbase_pe=10,
    use_coarse_geo=False,
)

fine_train = dict(
    lrate_k0=0, 
    ray_sampler='random',
)

