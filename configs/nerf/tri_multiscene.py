_base_ = '../tri_multiscene_default.py'

expname = '3conv_liif_scratch_coarse+fine' # '3MLP_liif_pretrain_down4_cosine_featunfold' #'rnd_liif_sum_sep_interp'
basedir = './logs/tri_dvgo_multiscene/nerf_synthetic/'

data = dict(
    down=4,
    task='',
    # datadir='./data/nerf_synthetic/',
    datadir='/home/hydeng/data/NeRF_data/nerf_synthetic',
    dataset_type='blender',
    white_bkgd=True,
    render_down=4,
    batch_size=1, 
    dataset='MultisceneBlenderDataset_v2',
    test_scenes=['hotdog', 'mic', 'lego']
)

coarse_train = dict(
    N_iters=5000, # could be larger
    N_rand=1024,
)

coarse_model_and_render = dict(
    n_scene=8, 
)

fine_train = dict(
    N_iters=0,
    N_rand=6144,
    lrate_k0=0, 
    lrate_map=0, #1e-4,
    lrate_encoder=1e-4,
    lrate_interp=0,
    lrate_interp_xy=5e-4,
    lrate_interp_yz=5e-4,
    lrate_interp_zx=5e-4,

    lrate_map_xy=5e-4,
    lrate_map_yz=5e-4,
    lrate_map_zx=5e-4,

    lrate_distillation_head=1e-4,

    lrate_decay=400,
    pg_scale=[20000, 60000, 10000, 12000],
    fixed_lr_idx=[], #[34, 49, 63],
    fixed_lr_idx_render = [34, 49, 63], 
    ray_sampler='random',

    dynamic_downsampling=True,
    dynamic_down=16,
    # skip_zero_grad_fields=[],
    weight_consistency=0.01, #100, 
    weight_cosine=0.01,
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    
    weight_distillation=0,
)

fine_model_and_render = dict(
    implicit_voxel_feat=True,
    feat_unfold=False,
    cell_decode=True,
    local_ensemble=True,
    use_coarse_geo=True,
    rgbnet_dim=64,
    name='edsr-baseline' , # 'resnet34', #
    posbase_pe=0,

    rgbnet_depth=5,

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

    mlp_map=False, 
    conv_map=True,
    closed_map=False,
    liif_state_dict='/home/hydeng/Documents/SR_NeRF/code/DirectVoxGO/pretrained/edsr-baseline-liif.pth',
    # liif_state_dict='/data/hydeng/SR_NeRF/liif/checkpoints/edsr-baseline-liif.pth',
    load_liif_sd=False,
    pretrained_state_dict='/home/hydeng/Documents/SR_NeRF/code/DirectVoxGO/pretrained/edsr-baseline.pth',
    compute_consistency=True,
    
    compute_cosine=True,

    n_mapping=3,

    use_anchor_liif=False,
    use_siren=False,

    stepsize=0.5,
)