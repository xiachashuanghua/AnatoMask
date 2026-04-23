import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from timm.layers import DropPath

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (D_, H_, W_) in enumerate(spatial_shapes):

        ref_d, ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                                                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

        ref_d = ref_d.reshape(-1)[None] /  D_
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] /  W_

        ref = torch.stack((ref_d, ref_x, ref_y), -1)   # D W H
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points



def deform_inputs(x):
    bs, c, d, h, w = x.shape
    spatial_shapes1 = torch.as_tensor([
        (d // 8,  h // 8,  w // 8),
        (d // 16, h // 16, w // 16),
        (d // 32, h // 32, w // 32),
    ], dtype=torch.long, device=x.device)
    level_start_index1 = torch.cat((
        spatial_shapes1.new_zeros(1),
        spatial_shapes1.prod(1).cumsum(0)[:-1]
    ))

    reference_points1 = get_reference_points([
        (d // 16, h // 16, w // 16)
    ], x.device)
    deform_inputs1 = [reference_points1, spatial_shapes1, level_start_index1]
    spatial_shapes2 = torch.as_tensor([
        (d // 16, h // 16, w // 16)
    ], dtype=torch.long, device=x.device)
    level_start_index2 = torch.cat((
        spatial_shapes2.new_zeros(1),
        spatial_shapes2.prod(1).cumsum(0)[:-1]
    ))
    reference_points2 = get_reference_points([
        (d // 8,  h // 8,  w // 8),
        (d // 16, h // 16, w // 16),
        (d // 32, h // 32, w // 32),
    ], x.device)
    deform_inputs2 = [reference_points2, spatial_shapes2, level_start_index2]
    return deform_inputs1, deform_inputs2



class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)   # 使用 3D depthwise conv
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        """
        x: [B, N, C], N = D * H * W
        """
        x = self.fc1(x)
        x = self.dwconv(x, D, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # 3D 深度可分离卷积
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1,
                                bias=True, groups=dim)
    def forward(self, x, D, H, W):


        B, N, C = x.shape
        # 这里的假设是 N 可以被 73 整除
        n = N // 73

        # 64n 对应尺度 (2D, 2H, 2W)
        x1 = x[:, 0:64 * n, :].transpose(1, 2).contiguous() \
             .view(B, C, D * 2, H * 2, W * 2)

        # 8n 对应尺度 (D, H, W)
        x2 = x[:, 64 * n:72 * n, :].transpose(1, 2).contiguous() \
             .view(B, C, D, H, W)

        # 1n 对应尺度 (D/2, H/2, W/2)
        x3 = x[:, 72 * n:, :].transpose(1, 2).contiguous() \
             .view(B, C, D // 2, H // 2, W // 2)

        # 分别做 3D depthwise conv
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)  # [B, 64n, C]
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)  # [B,  8n, C]
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)  # [B,   n, C]
        # 再拼回去
        x = torch.cat([x1, x2, x3], dim=1)  # [B, N, C]
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index,D, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query),D, H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2,D, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2],D=D, H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2],D=D, H=H, W=W)
        return x, c


class InteractionBlockWithCls(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2,D, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2],D=D, H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2],D=D, H=H, W=W)
        return x, c, cls


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, in_channels=1, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        # stem: 下采样 2 倍 + MaxPool 再下采样 2 倍 -> /4
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)   # /4
        )

        # /8
        self.conv2 = nn.Sequential(
            nn.Conv3d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        )
        # /16
        self.conv3 = nn.Sequential(
            nn.Conv3d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        # /32
        self.conv4 = nn.Sequential(
            nn.Conv3d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1 conv 映射到 embed_dim
        self.fc1 = nn.Conv3d(inplanes,       embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv3d(2 * inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv3d(4 * inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv3d(4 * inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        """
        x: [B, C, D, H, W]

        返回:
          c1: [B, embed_dim, D/4,  H/4,  W/4]   （保持为体素特征）
          c2: [B, N2, embed_dim]  （/8 展平为序列）
          c3: [B, N3, embed_dim]  （/16 展平为序列）
          c4: [B, N4, embed_dim]  （/32 展平为序列）
        """

        def _inner_forward(x):
            c1 = self.stem(x)   # /4
            c2 = self.conv2(c1) # /8
            c3 = self.conv3(c2) # /16
            c4 = self.conv4(c3) # /32

            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, D1, H1, W1 = c1.shape

            # c1 保持 [B, C, D, H, W] 形式给后面用
            # 其他三个展平成 [B, N, C] 作为 token 序列
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # /8
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # /16
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # /32

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)

        return outs
