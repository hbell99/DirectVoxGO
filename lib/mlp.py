from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF_MLP(nn.Module):

    def __init__(self, D=8, W=256, input_ch=99, input_ch_views=27, skips=[2]):
        """ 
        """
        super(NeRF_MLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        self.feature_linear = nn.Linear(W, W)
        self.density_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
        nn.init.constant_(self.rgb_linear.bias, 0)

    def forward(self, emb, viewemb):
        h = emb
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([emb, h], -1)

        density = self.density_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, viewemb], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)

        return rgb, density


class Mapping(nn.Module):
    def __init__(self, in_dim, out_dim=12, depth=1, width=64):
        super().__init__()
        self.feat_linears = nn.Sequential(
            nn.Linear(in_dim, width),  nn.ReLU(inplace=True),  
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth-2)
            ]
        )
        self.out_linear = nn.Linear(width, out_dim)
    
    def forward(self, feature, pose):
        # feature: [1, c, h, w]
        # pose: [1, 4, 4]
        # out: [1, out_dim, h, w] 
        feature = feature.permute(0, 2, 3, 1) # [1, h, w, c]
        _, h, w, c = feature.shape

        __, l, l = pose.shape
        assert __ == _
        pose = pose.reshape(_, 1, 1, -1)
        pose = pose.repeat(1, h, w, 1)

        feat = torch.cat([feature, pose], dim=-1)

        hid = self.feat_linears(feat)
        out = self.out_linear(hid)

        out = out.permute(0, 3, 1, 2)

        return out


class Interp_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=5):
        super(Interp_MLP, self).__init__()
        self.model = nn.Sequential(
            *[nn.Linear(in_dim, width), nn.ReLU(inplace=True)],
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth-2)
            ],
            nn.Linear(width, out_dim),
        )
    
    def forward(self, x):
        return self.model(x)

class Conv_Mapping(nn.Module):
    def __init__(self, in_dim, out_dim=12, depth=1, width=64):
        super().__init__()
        pass
    
    def forward(self, feature, pose):
        # feature: [1, c, h, w]
        # pose: [4, 4]
        # out: [1, out_dim, h, w] 
        _, c, h, w = feature.shape
        pose = pose.reshape(1, 16, 1, 1)
        pose_map = pose.repeat(1, 1, h, w)
        assert pose_map.shape[-2:] == feature.shape[-2:]

        pass
