import numpy as np
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
    def __init__(self, in_dim, out_dim=12, depth=1, width=64, dropout=0.1):
        super().__init__()
        self.feat_linears = nn.Sequential(
            nn.Linear(in_dim, width),  nn.ReLU(inplace=True),  
            *[
                nn.Sequential(nn.Linear(width, width), nn.Dropout(p=dropout), nn.ReLU(inplace=True))
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
    def __init__(self, in_dim, out_dim, width=128, depth=5, dropout=0.1):
        super(Interp_MLP, self).__init__()
        self.model = nn.Sequential(
            *[nn.Linear(in_dim, width), nn.ReLU(inplace=True)],
            *[
                nn.Sequential(nn.Linear(width, width), nn.Dropout(p=dropout), nn.ReLU(inplace=True))
                for _ in range(depth-2)
            ],
            nn.Linear(width, out_dim),
        )
    
    def forward(self, x):
        return self.model(x)



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), dropout=0.1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            m.append(nn.Dropout2d(p=dropout))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class Conv_Mapping(nn.Module):
    def __init__(self, in_dim, out_dim=12, kernel_size=3, n_resblocks=5):
        super().__init__()
        # self.feat_linears = nn.Sequential(
        #     nn.Linear(in_dim, width),  nn.ReLU(inplace=True),  
        #     *[
        #         nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
        #         for _ in range(depth-2)
        #     ]
        # )
        # self.out_linear = nn.Linear(width, out_dim)
        act = nn.ReLU(inplace=True)
        m_body = [
            ResBlock(
                default_conv, in_dim, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(in_dim, out_dim, kernel_size))

        self.body = nn.Sequential(*m_body)
    
    def forward(self, feature, pose):
        # feature: [1, c, h, w]
        # pose: [1, 4, 4]
        # out: [1, out_dim, h, w] 
        _, c, h, w = feature.shape

        __, l, l = pose.shape
        assert __ == _
        pose = pose.reshape(_, -1, 1, 1)
        pose = pose.repeat(1, 1, h, w)

        feat = torch.cat([feature, pose], dim=1)

        out = self.body(feat)
        return out

# class Conv_Mapping_pure(nn.Module):
#     def __init__(self, in_dim, out_dim=12, kernel_size=3, n_resblocks=3):
#         super().__init__()
        
#         act = nn.ReLU(inplace=True)
#         m_body = [
#             ResBlock(
#                 default_conv, in_dim, kernel_size, act=act
#             ) for _ in range(n_resblocks)
#         ]
#         m_body.append(default_conv(in_dim, out_dim, kernel_size))

#         self.body = nn.Sequential(*m_body)
    
#     def forward(self, feat):
#         # feature: [1, c, h, w]
#         # pose: [1, 4, 4]
#         # out: [1, out_dim, h, w] 

#         out = self.body(feat)
#         return out

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SirenRGB_net(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim):
        super().__init__()
        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, 3, is_last=True))

        self.rgb_net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.rgb_net(x)



# nn.Sequential(
#     nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
#     *[
#         nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
#         for _ in range(rgbnet_depth-2)
#     ],
#     nn.Linear(rgbnet_width, 3),
# )
# nn.init.constant_(rgbnet[-1].bias, 0)