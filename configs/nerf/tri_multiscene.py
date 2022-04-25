_base_ = '../tri_multiscene_default.py'

expname = 'debug' #'rnd_liif_sum_sep_interp'
basedir = './logs/tri_dvgo_multiscene/nerf_synthetic/'

data = dict(
    down=1,
    task='sr',
    datadir='/home/hydeng/data/NeRF_data/nerf_synthetic/',
    dataset_type='blender',
    white_bkgd=True,
    render_down=4,
    batch_size=1, 
)

coarse_train = dict(
    N_iters=0, # could be larger
)

# coarse_model_and_render = dict(
#     mask_cache_thres=7.5e-4, 
# )

fine_train = dict(
    N_iters=200000,
    N_rand=6144,
    lrate_k0=0, 
    lrate_map=1e-4,
    lrate_encoder=1e-4,
    lrate_interp=0,
    lrate_interp_xy=5e-4,
    lrate_interp_yz=5e-4,
    lrate_interp_zx=5e-4,
    lrate_rgbnet=5e-4,

    lrate_decay=400,
    pg_scale=[5000, 8000, 12000, 15000],
    fixed_lr_idx=[], #[34, 49, 63],
    ray_sampler='random',

    dynamic_downsampling=True,
    dynamic_down=16,
    # skip_zero_grad_fields=[],
)

fine_model_and_render = dict(
    implicit_voxel_feat=True,
    feat_unfold=False,
    cell_decode=True,
    local_ensemble=True,
    use_coarse_geo=False,
    rgbnet_dim=32,
    name='edsr-baseline' , # 'resnet34', #
    posbase_pe=0,

    rgbnet_depth=3,

    global_cell_decode=False,
    no_voxel_feat=False,
    cat_posemb=False,

    interp_width=128,
    interp_depth=5,

    map_depth=5,

    tri_aggregation='sum',
    liif=True,

    feat_pe=0,
    feat_fourier=False,
    n_scene=8, 
    pretrained_state_dict='/home/hydeng/Documents/SR_NeRF/code/DirectVoxGO/pretrained/edsr-baseline.pth',
)