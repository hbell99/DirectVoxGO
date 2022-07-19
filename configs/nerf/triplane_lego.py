_base_ = '../default.py'

expname = 'hotdog_ft_interp_mlp'
basedir = './logs/triplane_only_singlescene'

data = dict(
    datadir='/home/hydeng/data/NeRF_data/nerf_synthetic/hotdog',
    dataset_type='blender',
    white_bkgd=True,
)

coarse_train = dict(
    # N_iters=0,
    lrate_k0=0,
    lrate_density=0,
    lrate_k0_xy=5e-4,
    lrate_k0_yz=5e-4,
    lrate_k0_zx=5e-4,
)

coarse_model_and_render = dict(
    rgbnet_dim=12,
    posbase_pe=0,

    implicit_voxel_feat=False,
)


fine_model_and_render = dict(
    rgbnet_dim=12,
    posbase_pe=0,
    use_coarse_geo=True,
    bbox_thres=1e-2,
    mask_cache_thres=1e-2,

    implicit_voxel_feat=True,
    feat_unfold=False, 
    cell_decode=True,
    local_ensemble=True,

    interp_width=128,
    interp_depth=4,
    n_interp=1, 

    k0_ckpt_path='logs/triplane_only_singlescene/hotdog_w_coarse/k0_last.tar',
    rgbnet_ckpt_path='logs/triplane_only_singlescene/hotdog_w_coarse/rgbnet_last.tar',
)

fine_train = dict(
    # N_rand=4096, 
    # N_iters=40000,

    # lrate_rgbnet=0,

    lrate_density=0,
    
    lrate_k0=0,
    lrate_k0_xy=5e-4,
    lrate_k0_yz=5e-4,
    lrate_k0_zx=5e-4,
    
    lrate_interp=0,
    lrate_interp_xy=5e-4,
    lrate_interp_yz=0,
    lrate_interp_zx=0,

    ray_sampler='random',
    pg_scale=[1000, 2000, 3000, 4000],
)

