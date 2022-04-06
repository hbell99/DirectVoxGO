_base_ = '../tri_default.py'

expname = 'fix_edsr_cell'
basedir = './logs/tri_dvgo/nerf_synthetic/lego'

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
    # N_iters=40000,
    lrate_k0=0, 
    lrate_map=1e-4,
    lrate_encoder=1e-4,
    lrate_interp=1e-4,
    pg_scale=[],
    fixed_lr_idx=[34, 49, 63],
    ray_sampler='random',
    # skip_zero_grad_fields=[],
)

fine_model_and_render = dict(
    implicit_voxel_feat=False,
    feat_unfold=False,
    use_coarse_geo=True,
    cell_decode=True,
    # rgbnet_dim=32,
    name='edsr-baseline' , # 'resnet34', #
    posbase_pe=10,

    no_voxel_feat=False,
    cat_posemb=False,
)