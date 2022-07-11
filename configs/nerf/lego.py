_base_ = '../default.py'

expname = 'dvgo_down4'
basedir = './logs/nerf_synthetic'

data = dict(
    down=1,
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

# coarse_train = dict(
#     N_iters=2000,
# )

fine_model_and_render = dict(
    # posbase_pe=10,
    use_coarse_geo=False,
)

fine_train = dict(
    N_iters=15000,
    # lrate_k0=0, 
    # ray_sampler='random',
)

