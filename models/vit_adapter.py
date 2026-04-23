import logging
import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.modules.ms_deform_attn import MSDeformAttn
from timm.layers import trunc_normal_
from torch.nn.init import normal_

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from ops.modules.adapter_modules import (InteractionBlock, SpatialPriorModule,
                              deform_inputs)


class ViTAdapter(nn.Module):
    """
    把 Shanghai AI Lab 的 ViT-Adapter 思路，
    换成基于 MONAI 的 3D ViT 实现（不依赖 mmcv / mmseg）。
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        window_size: int = 0,
        window_block_indexes: tuple[int, ...] = (),
        # 下面是 ViTAdapter 特有的参数
        conv_inplane: int = 64,
        deform_num_heads: int = 12,
        n_points: int = 4,
        deform_ratio: float = 0.5,
        cffn_ratio: float = 0.25,
        with_cffn: bool = True,
        interaction_indexes=None,   # 类似 [[0,1,2,3],[4,5,6,7],...]
        use_extra_extractor: bool = True,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.img_size = (
            img_size if isinstance(img_size, Sequence) else (img_size,) * spatial_dims
        )
        self.patch_size = (
            patch_size
            if isinstance(patch_size, Sequence)
            else (patch_size,) * spatial_dims
        )
        grid_size = tuple(i // p for i, p in zip(img_size, patch_size))
        self.spatial_dims = spatial_dims

        # ---------------- ViT 主干（从 MONAI ViT 拆出来） ----------------
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            ws = window_size if i in window_block_indexes else 0
            blk = TransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                save_attn=save_attn,
                # 关键开关👇（你的 TransformerBlock/SABlock 已支持）
                window_size=ws,
                input_size=grid_size,
            )
            self.blocks.append(blk)
        self.norm = nn.LayerNorm(hidden_size)

        # 不做分类，只做 dense prediction backbone
        self.cls_token = None

        # ---------------- ViT-Adapter 部分 ----------------
        self.interaction_indexes = interaction_indexes or [
            list(range(num_layers))
        ]  # 默认一段

        self.embed_dim = hidden_size
        embed_dim = hidden_size


        self.spm = SpatialPriorModule(
            inplanes=conv_inplane,
            embed_dim=embed_dim,
            in_channels=in_channels,
            with_cp=False,
        )

        # level embedding（3 个尺度：/8,/16,/32）
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))


        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    deform_ratio=deform_ratio,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    extra_extractor=(
                        (True if i == len(self.interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                )
                for i in range(len(self.interaction_indexes))
            ]
        )

        # /8 → /4 的 3D 上采样
        self.up = nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2)

        # 每个尺度一个 3D Norm
        self.norm1 = nn.BatchNorm3d(embed_dim)
        self.norm2 = nn.BatchNorm3d(embed_dim)
        self.norm3 = nn.BatchNorm3d(embed_dim)
        self.norm4 = nn.BatchNorm3d(embed_dim)

        # 初始化
        self.apply(self._init_weights)
        nn.init.normal_(self.level_embed, std=0.02)

    # ---------------- init ----------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            fan_out = (
                m.kernel_size[0]
                * m.kernel_size[1]
                * m.kernel_size[2]
                * m.out_channels
            )
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    # ---------------- forward ----------------
    def forward(self, x):
        """
        x: [B, C, D, H, W]
        返回: 四个尺度特征 [f1(/4), f2(/8), f3(/16), f4(/32)]
        """

        B = x.size(0)

        # 1) 3D deformable attention 的 meta 信息
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # 2) CNN Spatial Prior
        c1, c2, c3, c4 = self.spm(x)
        # c1: [B, C, D/4, H/4, W/4]
        # c2,c3,c4: [B, Ni, C] (flatten 的 token)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  # [B, N2+N3+N4, C]

        # 3) ViT Patch Embedding（MONAI 版已经包含 pos_embed）
        tokens = self.patch_embedding(x)  # [B, N, C]
        # 根据 img_size / patch_size 反推 patch grid 大小
        D = self.img_size[0] // self.patch_size[0]
        H = self.img_size[1] // self.patch_size[1]
        W = self.img_size[2] // self.patch_size[2]
        N = D * H * W
        assert tokens.shape[1] == N, "tokens 数和 D*H*W 对不上，检查 img_size / patch_size"

        # 4) 交互：在某些 block 范围内插入 InteractionBlock3D
        x_vit = tokens
        outs = []
        start_block = 0
        for i, layer in enumerate(self.interactions):
            idxs = self.interaction_indexes[i]  # 比如 [0,1,2,3]
            # 先跑到这一段开始之前的 blocks
            for blk in self.blocks[start_block : idxs[0]]:
                x_vit = blk(x_vit)
            # 然后这一段 blocks 由 InteractionBlock3D 接管（内部再调用这段 blocks）
            x_vit, c = layer(
                x_vit,
                c,
                self.blocks[idxs[0] : idxs[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                D,
                H,
                W,
            )
            # 保存这一阶段的 ViT feature map
            outs.append(
                x_vit.transpose(1, 2)
                .view(B, self.embed_dim, D, H, W)
                .contiguous()
            )
            start_block = idxs[-1] + 1

        # 把剩余 blocks 跑完（如果有的话）
        for blk in self.blocks[start_block:]:
            x_vit = blk(x_vit)
        x_vit = self.norm(x_vit)

        # 5) 把 c 的 token 再拆回 3 个尺度的 feature map
        c2_len, c3_len, c4_len = c2.size(1), c3.size(1), c4.size(1)
        c2_tok = c[:, 0:c2_len, :]
        c3_tok = c[:, c2_len : c2_len + c3_len, :]
        c4_tok = c[:, c2_len + c3_len :, :]

        # 这里的 D,H,W 对应 ViT 主分辨率（/16），
        # 根据 SPM 你的设计决定每一级的 scale（下面是一个示例，你可以和 SPM 对齐调整）
        c2_map = (
            c2_tok.transpose(1, 2)
            .view(B, self.embed_dim, D * 2, H * 2, W * 2)
            .contiguous()
        )  # /8
        c3_map = (
            c3_tok.transpose(1, 2)
            .view(B, self.embed_dim, D, H, W)
            .contiguous()
        )  # /16
        c4_map = (
            c4_tok.transpose(1, 2)
            .view(B, self.embed_dim, D // 2, H // 2, W // 2)
            .contiguous()
        )  # /32

        # c1 是 /4，c2_map 是 /8，用 up 把 /8 提到 /4 再相加
        c1_24 = self.up(c2_map) + c1  # ~ /4 -> 大约 24^3

        # 6) 选择是否叠加 ViT feature（和原 ViT-Adapter 一样）
        if len(outs) >= 4:
            x1, x2, x3, x4 = outs[-4:]
        else:
            # 不够 4 层就简单复用最后一层
            last = outs[-1]
            x1 = x2 = x3 = x4 = last

        x1_48 = F.interpolate(
            x1, scale_factor=8, mode="trilinear", align_corners=False
        )  # 6 * 8 = 48
        x2_24 = F.interpolate(
            x2, scale_factor=4, mode="trilinear", align_corners=False
        )  # 6 * 4 = 24
        x3_12 = F.interpolate(
            x3, scale_factor=2, mode="trilinear", align_corners=False
        )  # 6 * 2 = 12
        x4_6 = x4  # 6 * 1 = 6

        # --- 再把 CNN 先验特征也对齐到同样的分辨率 ---
        c1_48 = F.interpolate(
            c1_24, scale_factor=2, mode="trilinear", align_corners=False
        )  # 24 -> 48
        c2_24 = F.interpolate(
            c2_map, scale_factor=2, mode="trilinear", align_corners=False
        )   # 12 -> 24
        c3_12 = F.interpolate(
            c3_map, scale_factor=2, mode="trilinear", align_corners=False
        )   #  6 -> 12
        c4_6 = F.interpolate(
            c4_map, scale_factor=2, mode="trilinear", align_corners=False
        )    #  3 -> 6

        # 7) 融合 + Norm，得到最终四层输出
        f1 = self.norm1(c1_48 + x1_48)  # [B, C, 48, 48, 48]
        f2 = self.norm2(c2_24 + x2_24)  # [B, C, 24, 24, 24]
        f3 = self.norm3(c3_12 + x3_12)  # [B, C, 12, 12, 12]
        f4 = self.norm4(c4_6 + x4_6)    # [B, C,  6,  6,  6]

        return [f1, f2, f3, f4]
