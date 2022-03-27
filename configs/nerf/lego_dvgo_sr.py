_base_ = '../default.py'

expname = 'debug'
basedir = './logs/implicit_voxel_feat/nerf_synthetic/lego'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=5000,
)

# fine_train = dict(
#     N_iters=0,
# )

fine_model_and_render = dict(
    implicit_voxel_feat=False,
    # rgbnet_dim=4,
    # feat_unfold=True
    rgbnet_dim=12,
    feat_unfold=False
)