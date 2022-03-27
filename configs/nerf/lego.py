_base_ = '../default.py'

expname = 'dvgo_lego_fineposemb'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/lego',
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
    # N_iters=40000,
    lrate_k0=0, 
    # lrate_decay=8,
    # pg_scale=[5000, 10000, 20000, 30000],
)

