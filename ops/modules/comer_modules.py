import logging
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
from ops.modules import MSDeformAttn
from timm.layers import DropPath
import torch.utils.checkpoint as cp

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

def deform_inputs_only_one(x, d, h, w):
    """
    3D deformable attention 输入构造。
    参数:
        x: Tensor[B, C, D, H, W]
        d, h, w: 输入的深度/高度/宽度
    返回:
        [reference_points, spatial_shapes, level_start_index]
    """
    device = x.device
    spatial_shapes = torch.as_tensor([
        (d // 8,  h // 8,  w // 8),
        (d // 16, h // 16, w // 16),
        (d // 32, h // 32, w // 32)
    ], dtype=torch.long, device=device)

    level_start_index = torch.cat((
        spatial_shapes.new_zeros(1),
        spatial_shapes.prod(1).cumsum(0)[:-1]
    ))
    reference_points = get_reference_points([
        (d // 8,  h // 8,  w // 8),
        (d // 16, h // 16, w // 16),
        (d // 32, h // 32, w // 32)
    ], device=device)
    return [reference_points, spatial_shapes, level_start_index]


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


class MRFP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,D, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, D,H, W)
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
    

class MultiDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        dim1 = dim               # 总通道数
        dim_half = dim // 2      # 每个分支用一半通道

        # 每个尺度上：一半通道用 3x3x3，一半用 5x5x5
        self.dwconv1 = nn.Conv3d(dim_half, dim_half, 3, 1, 1, bias=True, groups=dim_half)
        self.dwconv2 = nn.Conv3d(dim_half, dim_half, 5, 1, 2, bias=True, groups=dim_half)

        self.dwconv3 = nn.Conv3d(dim_half, dim_half, 3, 1, 1, bias=True, groups=dim_half)
        self.dwconv4 = nn.Conv3d(dim_half, dim_half, 5, 1, 2, bias=True, groups=dim_half)

        self.dwconv5 = nn.Conv3d(dim_half, dim_half, 3, 1, 1, bias=True, groups=dim_half)
        self.dwconv6 = nn.Conv3d(dim_half, dim_half, 5, 1, 2, bias=True, groups=dim_half)

        self.act1 = nn.GELU()
        self.bn1  = nn.GroupNorm(num_groups=32, num_channels=dim1) #nn.BatchNorm3d(dim1)

        self.act2 = nn.GELU()
        self.bn2  = nn.GroupNorm(num_groups=32, num_channels=dim1) 

        self.act3 = nn.GELU()
        self.bn3  = nn.GroupNorm(num_groups=32, num_channels=dim1) 

    def forward(self, x, D, H, W):
        """
        x: [B, N, C]，N 假设对应三个尺度拼接：
           前 64n -> (2D, 2H, 2W)
           中  8n -> ( D,  H,  W)
           后  1n -> (D/2, H/2, W/2)
           N = 73n
        """
        B, N, C = x.shape
        n = N // 73

        # ----------- 拆成三种尺度 -----------
        # 大尺度: (2D, 2H, 2W)，对应 64n 个 token
        x1 = x[:, 0:64 * n, :] \
              .transpose(1, 2).contiguous() \
              .view(B, C, D * 2, H * 2, W * 2)

        # 中尺度: (D, H, W)，对应 8n 个 token
        x2 = x[:, 64 * n:72 * n, :] \
              .transpose(1, 2).contiguous() \
              .view(B, C, D, H, W)

        # 小尺度: (D/2, H/2, W/2)，对应 1n 个 token
        x3 = x[:, 72 * n:, :] \
              .transpose(1, 2).contiguous() \
              .view(B, C, D // 2, H // 2, W // 2)

        # ----------- 每个尺度上做 双分支 DWConv3D (3x3x3 + 5x5x5) -----------

        # x1: (2D,2H,2W)
        x11, x12 = x1[:, :C//2, ...], x1[:, C//2:, ...]
        x11 = self.dwconv1(x11)
        x12 = self.dwconv2(x12)
        x1 = torch.cat([x11, x12], dim=1)           # [B, C, 2D,2H,2W]
        x1 = self.act1(self.bn1(x1))               # BN+GELU
        x1 = x1.flatten(2).transpose(1, 2)         # [B, 64n, C]

        # x2: (D,H,W)
        x21, x22 = x2[:, :C//2, ...], x2[:, C//2:, ...]
        x21 = self.dwconv3(x21)
        x22 = self.dwconv4(x22)
        x2 = torch.cat([x21, x22], dim=1)
        x2 = self.act2(self.bn2(x2))
        x2 = x2.flatten(2).transpose(1, 2)         # [B, 8n, C]

        # x3: (D/2,H/2,W/2)
        x31, x32 = x3[:, :C//2, ...], x3[:, C//2:, ...]
        x31 = self.dwconv5(x31)
        x32 = self.dwconv6(x32)
        x3 = torch.cat([x31, x32], dim=1)
        x3 = self.act3(self.bn3(x3))
        x3 = x3.flatten(2).transpose(1, 2)         # [B, n, C]

        # ----------- 拼回 token 序列 -----------
        x = torch.cat([x1, x2, x3], dim=1)         # [B, N, C]
        return x

class MultiscaleExtractor(nn.Module):
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

    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, D,H, W):
        
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


class CTI_toC(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        # if with_cffn:
        #     self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        #     self.ffn_norm = norm_layer(dim)
        #     self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(dim=dim, n_levels=3, num_heads=num_heads,
                                n_points=n_points, norm_layer=norm_layer, 
                                deform_ratio=deform_ratio, with_cffn=with_cffn,
                                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, 
                                with_cp=with_cp)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index,D, H, W):
        
        def _inner_forward(query, feat, D,H, W):
            B, N, C = query.shape
            # 3D 版切分：高分辨率(2D,2H,2W)、中分辨率(D,H,W)、低分辨率(D/2,H/2,W/2)
            n_mid  = D * H * W          # 中分辨率 token 数，等于 feat 的长度
            n_high = n_mid * 8          # 高分辨率 token 数

            x1 = query[:, :n_high, :].contiguous()               # high-res
            x2 = query[:, n_high:n_high + n_mid, :].contiguous() # mid-res
            x3 = query[:, n_high + n_mid:, :].contiguous()       # low-res

            # 这里 x2 和 feat 的长度一样，才可以加
            x2 = x2 + feat
            query = torch.cat([x1, x2, x3], dim=1)

            # if self.with_cffn:
            #     query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W)) 

            if self.cnn_feature_interaction:               
                deform_input = deform_inputs_only_one(
                    query, D * 16, H * 16, W * 16
                )
                query = self.cfinter(query=self.query_norm(query), reference_points=deform_input[0],
                          feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                          level_start_index=deform_input[2],
                          D=D,H=H, W=W)               
            
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, D,H, W)
        else:
            query = _inner_forward(query, feat,D, H, W)
        
        return query

class Extractor_CTI(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(dim=dim, n_levels=3, num_heads=num_heads,
                                n_points=n_points, norm_layer=norm_layer, 
                                deform_ratio=deform_ratio, with_cffn=with_cffn,
                                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, 
                                with_cp=with_cp)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, D,H, W):
        
        def _inner_forward(query, feat,D, H, W):
            B, N, C = query.shape
            n = N // 21
            # 3D 版切分：高分辨率(2D,2H,2W)、中分辨率(D,H,W)、低分辨率(D/2,H/2,W/2)
            n_mid  = D * H * W          # 中分辨率 token 数，等于 feat 的长度
            n_high = n_mid * 8          # 高分辨率 token 数

            x1 = query[:, :n_high, :].contiguous()               # high-res
            x2 = query[:, n_high:n_high + n_mid, :].contiguous() # mid-res
            x3 = query[:, n_high + n_mid:, :].contiguous()       # low-res

            # 这里 x2 和 feat 的长度一样，才可以加
            x2 = x2 + feat
            query = torch.cat([x1, x2, x3], dim=1)

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query),D, H, W)) 

            if self.cnn_feature_interaction:               
                deform_input = deform_inputs_only_one(
                    query, D * 16, H * 16, W * 16
                )
                query = self.cfinter(query=self.query_norm(query), reference_points=deform_input[0],
                          feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                          level_start_index=deform_input[2],
                          D=D,H=H, W=W)               
            
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat,D, H, W)
        else:
            query = _inner_forward(query, feat,D, H, W)
        
        return query



class CTI_toV(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, drop=0., drop_path=0., cffn_ratio=0.25):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
       
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index,D, H, W):
        
        def _inner_forward(query, feat, D,H, W):
            B, N, C = feat.shape
            c1 = self.attn(self.query_norm(feat), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)

            c1 = c1 + self.drop_path(self.ffn(self.ffn_norm(c1),D, H, W)) 

            n_mid = D * H * W               # 中分辨率 token 数
            n_high = n_mid * 8              # 高分辨率 token 数 (2D,2H,2W)
            # 剩下的就当做低分辨率
            c_select1 = c1[:, :n_high, :]              # 高分辨率 token
            c_select2 = c1[:, n_high:n_high + n_mid, :]# 中分辨率 token
            c_select3 = c1[:, n_high + n_mid:, :]      # 低分辨率 token
            # ==== 高分辨率： (2D,2H,2W) -> 插值到 (D,H,W) ====
            # [B, n_high, C] -> [B,C,2D,2H,2W]
            c1_vol = c_select1.permute(0, 2, 1).reshape(B, C, D * 2, H * 2, W * 2)
            c1_vol = F.interpolate(
                c1_vol,
                scale_factor=0.5,   # (2D,2H,2W) -> (D,H,W)
                mode="trilinear",
                align_corners=False,
            )
            c_select1 = c1_vol.flatten(2).permute(0, 2, 1)  # [B, D*H*W, C]

            # ==== 低分辨率： (D/2,H/2,W/2) -> 插值到 (D,H,W) ====
            # 期望剩余 token 数是 (D/2*H/2*W/2) = n_mid / 8
            # 如果不满足就会 reshape 报错，等价于帮你检查维度是否匹配
            d_low = D // 2
            h_low = H // 2
            w_low = W // 2
            if c_select3.size(1) != d_low * h_low * w_low:
                raise RuntimeError(
                    f"CTI_toV: 低分辨率 token 数不匹配，"
                    f"got {c_select3.size(1)}, expected {d_low*h_low*w_low}"
                )
            c3_vol = c_select3.permute(0, 2, 1).reshape(B, C, d_low, h_low, w_low)
            c3_vol = F.interpolate(
                c3_vol,
                scale_factor=2.0,   # (D/2,H/2,W/2) -> (D,H,W)
                mode="trilinear",
                align_corners=False,
            )
            c_select3 = c3_vol.flatten(2).permute(0, 2, 1)  # [B, D*H*W, C]
            # x = x + c_select1 + c_select2 + c_select3

            return query + self.gamma * (c_select1 + c_select2 + c_select3)
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat,D, H, W)
        else:
            query = _inner_forward(query, feat, D,H, W)
        
        return query


class CTIBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_CTI=False, with_cp=False, 
                 use_CTI_toV=True, 
                 use_CTI_toC=True,
                 dim_ratio=6.0,
                 cnn_feature_interaction=False):
        super().__init__()

        if use_CTI_toV:
            self.cti_tov = CTI_toV(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp, drop=drop, drop_path=drop_path, cffn_ratio=cffn_ratio)
        if use_CTI_toC:
            self.cti_toc = CTI_toC(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp,
                                   cnn_feature_interaction=cnn_feature_interaction)
        
        if extra_CTI:
            self.extra_CTIs = nn.Sequential(*[
                Extractor_CTI(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp,
                                   cnn_feature_interaction=cnn_feature_interaction)
                for _ in range(4)
            ])

        else:
            self.extra_CTIs = None
        
        self.use_CTI_toV = use_CTI_toV
        self.use_CTI_toC = use_CTI_toC

        self.mrfp = MRFP(dim, hidden_features=int(dim * dim_ratio))

    
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2,D, H, W):
        B, N, C = x.shape
        deform_inputs = deform_inputs_only_one(x,D * 16, H*16, W*16)
        if self.use_CTI_toV:
            c = self.mrfp(c, D,H, W)
            n_mid = D * H * W
            n_high = n_mid * 8

            c_select1 = c[:, :n_high, :].contiguous()
            c_select2 = c[:, n_high : n_high + n_mid, :].contiguous()
            c_select3 = c[:, n_high + n_mid :, :].contiguous()
            c = torch.cat([c_select1, c_select2 + x, c_select3], dim=1)

            x = self.cti_tov(query=x, reference_points=deform_inputs[0],
                          feat=c, spatial_shapes=deform_inputs[1],
                          level_start_index=deform_inputs[2],D=D, H=H, W=W)

        for idx, blk in enumerate(blocks):
            x = blk(x)

        if self.use_CTI_toC:
            c = self.cti_toc(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2],D=D, H=H, W=W)
                           
        if self.extra_CTIs is not None:
            for cti in self.extra_CTIs:
                c = cti(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2],D=D, H=H, W=W)
        return x, c


class CNN(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384,in_channels=1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=inplanes),
            #nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=inplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)   # /4
        )

        # /8
        self.conv2 = nn.Sequential(
            nn.Conv3d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.SyncBatchNorm(2 * inplanes),
            nn.GroupNorm(num_groups=32, num_channels=2*inplanes),
            nn.ReLU(inplace=True)
        )
        # /16
        self.conv3 = nn.Sequential(
            nn.Conv3d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.SyncBatchNorm(4 * inplanes),
            nn.GroupNorm(num_groups=32, num_channels=4* inplanes),
            nn.ReLU(inplace=True)
        )
        # /32
        self.conv4 = nn.Sequential(
            nn.Conv3d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.SyncBatchNorm(4 * inplanes),
            nn.GroupNorm(num_groups=32, num_channels=4* inplanes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1 conv 映射到 embed_dim
        self.fc1 = nn.Conv3d(inplanes,       embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv3d(2 * inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv3d(4 * inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv3d(4 * inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ ,_= c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4
