_base_ = '../sr_default.py'

expname = 'coarse_maskcache'
basedir = './logs/sr_dvgo/nerf_synthetic/lego'

data = dict(
    down=4,
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

# coarse_train = dict(
#     N_iters=0,
# )

fine_train = dict(
    N_iters=40000,
    lrate_k0=0, 
    pg_scale=[],
    # ray_sampler='random',
)

# fine_model_and_render = dict(
#     use_coarse_geo=True,
# )