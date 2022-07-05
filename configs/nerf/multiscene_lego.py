_base_ = '../multiscene_default.py'

expname = 'vox_sum'
basedir = './logs/multiscene_dvgo/lego/liif'

data = dict(
    down=1,
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
    render_down=4,
)

# coarse_train = dict(
#     N_iters=0, # could be larger
# )

# coarse_model_and_render = dict(
#     mask_cache_thres=7.5e-4, 
# )

fine_train = dict(
    N_iters=200000,
    N_rand=2048,
    lrate_k0=0, 
    lrate_map=5e-4,
    lrate_encoder=1e-4,
    lrate_interp=5e-4,
    lrate_rgbnet=5e-4,

    lrate_decay=100,
    fixed_lr_idx=[34, 49, 63],
    ray_sampler='random',
    # pg_scale=[],

    dynamic_downsampling=True,
    dynamic_down=16,
)

fine_model_and_render = dict(
    # num_voxels=1024000,
    # num_voxels_base=1024000,
    implicit_voxel_feat=True,
    feat_unfold=False,
    cell_decode=True,
    local_ensemble=True,
    use_coarse_geo=False,
    name='edsr-baseline',
    posbase_pe=0,

    global_cell_decode=False,
    no_voxel_feat=False,
    cat_posemb=False,
    interp_width=128,
    interp_depth=5,

    map_depth=5,

    # rgbnet_dim=32,
    rgbnet_width=256,
    rgbnet_depth=8,
    skips=[4],
    liif=True,

    # tri_aggregation='sum',
    
    use_mipnerf_density=True,
    # stepsize=1,
)