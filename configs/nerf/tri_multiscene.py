_base_ = '../tri_multiscene_default.py'

expname = '1resconv_newrgb_cell_liif_d_o_lego'# '3conv_liif_scratch_coarse+fine' 
basedir = './logs/tri_dvgo_multiscene_v1/nerf_synthetic/'

data = dict(
    down=4,
    task='',
    datadir='./data/nerf_synthetic/',
    # datadir='/home/hydeng/data/NeRF_data/nerf_synthetic',
    dataset_type='blender',
    white_bkgd=True,
    render_down=4,
    batch_size=1, 
    dataset='MultisceneBlenderDataset_v2',
    test_scenes=['lego']
)

coarse_train = dict(
    N_iters=10000, # could be larger
    N_rand=8192,
)

coarse_model_and_render = dict(
    n_scene=1, 
    bbox_thres=1e-3, 
    mask_cache_thres=1e-3, 
)

fine_train = dict(
    N_iters=50000,
    N_rand=4096,
    lrate_k0=0, 
    lrate_map=5e-4, #1e-4,
    lrate_encoder=1e-4,
    lrate_interp=0,
    lrate_interp_xy=5e-4,
    lrate_interp_yz=5e-4,
    lrate_interp_zx=5e-4,

    lrate_map_xy=5e-4,
    lrate_map_yz=5e-4,
    lrate_map_zx=5e-4,

    lrate_nl_block=1e-4, 
    lrate_rgbnet=1e-3, 

    lrate_distillation_head=5e-4,

    lrate_decay=1000,
    pg_scale=[1000, 2000, 3000, 5000],
    fixed_lr_idx=[], #[34, 49, 63],
    fixed_lr_idx_render=[34, 49, 63], # [61, 95, 46]
    ray_sampler='random',

    dynamic_downsampling=True,
    dynamic_down=4,
    # skip_zero_grad_fields=[],
    weight_entropy_last=0.001,
    weight_rgbper=0, #0.01,
    
    weight_distillation=0,
    weight_consistency=0, #100, 
    weight_cosine=0,
)

fine_model_and_render = dict(
    implicit_voxel_feat=True,
    feat_unfold=False,
    cell_decode=True,
    local_ensemble=True,
    use_coarse_geo=True,
    rgbnet_dim=64,
    name='edsr-baseline', # 'resnet34', # 'resnet34', #
    posbase_pe=5,

    rgbnet_depth=3,

    global_cell_decode=False,
    no_voxel_feat=False,
    cat_posemb=True,

    interp_width=128,
    interp_depth=3,

    map_depth=5,

    tri_aggregation='sum',
    liif=True,

    feat_pe=0,
    feat_fourier=False,
    n_scene=1, 

    mlp_map=False, 
    conv_map=True,
    closed_map=False,
    # liif_state_dict='/home/hydeng/Documents/SR_NeRF/code/DirectVoxGO/pretrained/edsr-baseline-liif.pth',
    liif_state_dict='/data/hydeng/SR_NeRF/liif/checkpoints/edsr-baseline-liif.pth',
    load_liif_sd=False,
    # pretrained_state_dict='/home/hydeng/Documents/SR_NeRF/code/DirectVoxGO/pretrained/edsr-baseline.pth',
    compute_consistency=True,
    
    compute_cosine=True,

    n_mapping=1,

    n_interp=3, 

    use_anchor_liif=False,
    use_siren=False,
    use_nl=False,
    use_liif_attn=False,

    stepsize=0.25,

    cosine_v1=False,
    cosine_v2=False,

    world_bound_scale=1.05, # 1.05
    # bbox_thres=1e-4, 
    # mask_cache_thres=1e-4, 
)