_base_ = '../default.py'

expname = 'dvgo_ship'
basedir = './logs/nerf_synthetic'

data = dict(
    down=4,
    datadir='./data/nerf_synthetic/ship',
    dataset_type='blender',
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=0,
)

fine_model_and_render = dict(
    # posbase_pe=10,
    use_coarse_geo=False,
)

fine_train = dict(
    N_iters=10000,
    # lrate_k0=0, 
    # ray_sampler='random',
)

