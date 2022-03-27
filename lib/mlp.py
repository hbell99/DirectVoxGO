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