import torch
import torch.nn as nn
import math
from monai.networks.nets import ViT
import torch.nn.functional as F
def seq_to_grid3d(x_seq, grid):  # x_seq: (B, N, C), grid=(D,H,W)
    B, N, C = x_seq.shape
    D,H,W = grid
    assert N in (D*H*W, D*H*W+1), f"N={N}与网格不匹配"
    if N == D*H*W + 1:  # 去掉 CLS
        x_seq = x_seq[:, 1:, :]
    x = x_seq.view(B, D, H, W, C).permute(0,4,1,2,3).contiguous()  # (B,C,D,H,W)
    return x
class SimplePyramidHead3D(nn.Module):
    """
    不依赖 backbone，仅把单尺度 (B,C,D,H,W) 变成多尺度金字塔。
    支持可选 top_block：在最小分辨率层（最后一层）上继续下采样 num_levels 次，生成 p(k+1), p(k+2), ...
    """
    def __init__(self, in_channels, out_channels, scale_factors, norm="gn", top_block: nn.Module | None = None):
        super().__init__()
        self.scale_factors = scale_factors
        self.top_block = top_block

        self.stages = nn.ModuleList()
        for s in scale_factors:
            layers = []
            out_dim = in_channels
            if s == 4.0:
                layers += [nn.ConvTranspose3d(in_channels, in_channels // 2, 2, 2),
                           nn.GroupNorm(1, in_channels // 2), nn.GELU(),
                           nn.ConvTranspose3d(in_channels // 2, in_channels // 4, 2, 2)]
                out_dim = in_channels // 4
            elif s == 2.0:
                layers += [nn.ConvTranspose3d(in_channels, in_channels // 2, 2, 2)]
                out_dim = in_channels // 2
            elif s == 1.0:
                pass
            elif s == 0.5:
                layers += [nn.MaxPool3d(2, 2)]
            elif s == 8.0:
                layers += [
                    nn.ConvTranspose3d(in_channels, in_channels // 2, 2, 2),
                    nn.GroupNorm(1, in_channels // 2), nn.GELU(),
                    nn.ConvTranspose3d(in_channels // 2, in_channels // 4, 2, 2),
                    nn.GroupNorm(1, in_channels // 4), nn.GELU(),
                    nn.ConvTranspose3d(in_channels // 4, in_channels // 8, 2, 2),
                ]
                out_dim = in_channels // 8
            else:
                raise NotImplementedError(s)

            layers += [nn.Conv3d(out_dim, out_channels, 1, bias=False),
                       nn.GroupNorm(1, out_channels),
                       nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                       nn.GroupNorm(1, out_channels)]
            self.stages.append(nn.Sequential(*layers))

        self.out_names = None  # 运行时设置

    def forward(self, feat, in_stride, scale_factors=None):
        """
        feat: (B,C,D,H,W)
        in_stride: 输入特征相对于原图的步幅（一般等于 patch_size）
        """
        if scale_factors is None:
            scale_factors = self.scale_factors

        # 主金字塔分支
        outs = [stage(feat) for stage in self.stages]

        # 基于 stride 命名
        strides = [int(in_stride / s) for s in scale_factors]
        for i in range(1, len(strides)):
            assert strides[i] == 2 * strides[i - 1], f"strides 非连续: {strides}"

        names = [f"p{int(math.log2(s))}" for s in strides]  # 与 outs 一一对应
        out_dict = {name: o for name, o in zip(names, outs)}

        # 可选：top_block 继续下采样（默认从最后一层，也就是 names[-1]）
        if self.top_block is not None:
            src = outs[-1]  # 以最小分辨率层作为输入（例如 p5）
            extra = self.top_block(src)  # 期待返回 list[Tensor]，长度 == num_levels
            assert isinstance(extra, (list, tuple)) and len(extra) == getattr(self.top_block, "num_levels", len(extra)), \
                "top_block 需要返回长度为 num_levels 的 list[Tensor]"
            last_stage = int(math.log2(strides[-1]))  # 例如 p5 -> stage=5
            for i, t in enumerate(extra, start=1):
                out_dict[f"p{last_stage + i}"] = t

        return out_dict

# —— 一个3D的 top_block 示例：从 p5 生成 p6 ——
class LastLevelMaxPool3D(nn.Module):
    """
    从最后层（如 p5）再下采样一倍得到 p6（stride×2）
    """
    def __init__(self):
        super().__init__()
        self.num_levels = 1

    def forward(self, x):
        # x: (N, C, D, H, W) -> (N, C, D/2, H/2, W/2)
        return [F.max_pool3d(x, kernel_size=1, stride=2, padding=0)]
if __name__ == "__main__":
    model = ViT(
        in_channels=1,
        img_size=96,
        patch_size=16,
        spatial_dims=3,
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        proj_type="conv",
        classification=False,
        dropout_rate=0,
        qkv_bias=False,
        save_attn=False,
        window_size = 7,
        window_block_indexes= ( 0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,),

    )
    fpn=SimplePyramidHead3D(768, 256, [8.0, 4.0, 2.0, 1.0, 0.5],)
    input_tensor = torch.randn(1, 1, 96, 96, 96)
    out,hidden_emd = model(input_tensor)
    print("out shape:")
    print(out.shape)
    print("hidden_emd shape:")
    for i in hidden_emd:
        print(i.shape)
    fpn_inputs = seq_to_grid3d(out, (6,6,6))
    print("fpn_inputs shape:")
    print(fpn_inputs.shape)
    fpn_outputs = fpn(fpn_inputs, 16, [8.0, 4.0, 2.0, 1.0, 0.5])
    print("fpn_outputs shape:")
    for name, f in fpn_outputs.items():
        print(f"{name:>4} : {tuple(f.shape)}")
