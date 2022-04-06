from turtle import forward
import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, views_dim, depth, width):
        super().__init__()
        
        self.feat_linears = nn.Sequential(
            nn.Linear(in_dim, width),  nn.ReLU(inplace=True),  
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth-2)
            ]
        )
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.Sequential(
            nn.Linear(views_dim + width, width // 2),
            nn.ReLU(inplace=True)
        )
        
        self.mid_linear = nn.Linear(width, width)
        self.density_linear = nn.Linear(width, 1)
        self.rgb_linear = nn.Linear(width // 2, 3)

    def forward(self, feat, views_emb):
        h = self.feat_linears(feat)

        density = self.density_linear(h)

        feature = self.mid_linear(h)
        h = torch.cat([feature, views_emb], -1)
    
        h = self.views_linears(h)

        rgb = self.rgb_linear(h)

        # density and rgb no activation !!!
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
        # pose: [4, 4]
        # out: [1, out_dim, h, w] 
        feature = feature.permute(0, 2, 3, 1) # [1, h, w, c]
        _, h, w, c = feature.shape

        pose = pose.reshape(-1)
        pose = pose.repeat(_, h, w, 1)

        feat = torch.cat([feature, pose], dim=-1)

        hid = self.feat_linears(feat)
        out = self.out_linear(hid)

        out = out.permute(0, 3, 1, 2)

        return out


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
