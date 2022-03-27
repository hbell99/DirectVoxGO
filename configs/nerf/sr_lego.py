_base_ = '../sr_default.py'

expname = 'debug'
basedir = './logs/sr_dvgo/nerf_synthetic/lego'

edsr_model = dict(
    name='baseline',
    n_resblocks=16, 
    n_feats=64, 
    res_scale=1,
    scale=2, 
    no_upsampling=True, 
    rgb_range=1
)

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=5000,
)

fine_train = dict(
    N_rand=2048,
    lrate_encoder=0,
    lrate_net=1e-5,
    N_iters=50000,
    normalize_lr=True,
    warmup_iters=1000
)

fine_model_and_render = dict(
    num_voxels=1024000,
    num_voxels_base=1024000,
    implicit_voxel_feat=False,
    feat_unfold=False,
    mask_cache_thres=1e-3,
    weight_entropy_last=0,
    weight_rgbper=0,
    net_depth=5,
    add_posemb=True,
    fast_color_thres=1e-4, # 1e-4
    use_coarse_geo=True,
    stepsize=0.5,
)