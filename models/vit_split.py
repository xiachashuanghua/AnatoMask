import torch
import torch.nn.functional as F
import itertools
import math
import torch.nn as nn
import torch.distributed as dist
from functools import partial
import copy
from models.vit import ViT
from tvdcn.ops import DeformConv3d
###Hack torch to load DINOv2####
from torch import Tensor
import torch._tensor
try:
    torch._tensor._rebuild_from_type_v2
except AttributeError:
    def _rebuild_from_type_v2(func, new_type, args, state):
        ret = func(*args)
        if type(ret) is not new_type:
            ret = ret.as_subclass(new_type)
        # Tensor does define __setstate__ even though it doesn't define
        # __getstate__. So only use __setstate__ if it is NOT the one defined
        # on Tensor
        if (
            getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
            is not Tensor.__setstate__
        ):
            ret.__setstate__(state)
        else:
            ret = torch._utils._set_obj_state(ret, state)
        return ret

    torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2

import torch._utils
try:
    torch._utils._set_obj_state
except AttributeError:
    def _set_obj_state(obj, state):
        if isinstance(state, tuple):
            if not len(state) == 2:
                raise RuntimeError(f"Invalid serialized state: {state}")
            dict_state = state[0]
            slots_state = state[1]
        else:
            dict_state = state
            slots_state = None

        # Starting with Python 3.11, the __dict__ attribute is lazily created
        # and is serialized as None when not needed.
        if dict_state:
            for k, v in dict_state.items():
                setattr(obj, k, v)
    
        if slots_state:
            for k, v in slots_state.items():
                setattr(obj, k, v)
        return obj

    torch._utils._set_obj_state = _set_obj_state




class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        return pad_left, pad_right

    def forward(self, x):
        # x.shape = (N, C, D, H, W)
        D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]

        pads = list(
            itertools.chain.from_iterable(
                self._get_pad(sz) for sz in (D, H, W)
            )
        )

        # F.pad expects reverse order: (W_l, W_r, H_l, H_r, D_l, D_r)
        pads = pads[::-1]

        return F.pad(x, pads)
class DeformableConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConvNet, self).__init__()
        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kD, kH, kW = kernel_size
        self.kernel_volume = kD * kH * kW
        num_offset_channels = 3 * kD * kH * kW
        self.offsets = nn.Conv3d(in_channels, num_offset_channels, kernel_size=kernel_size, padding=padding)
        self.deform_conv = DeformConv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
   
    def forward(self, x):
        offsets = self.offsets(x)
        B, _, D, H, W = offsets.shape

        # 关键：显式构造一个 mask，而不是让它默认为 None
        # mask 通道数 = kD * kH * kW，空间尺寸和 offset 一样
        mask = x.new_zeros(B, self.kernel_volume, D, H, W)

        # 显式传入 mask，绕开 mask=None 引发的 autograd bug
        out = self.deform_conv(x, offsets, mask)
        return x
class LearnableGate(torch.nn.Module):
    def __init__(self, n, k, out_num, temperature):
        super(LearnableGate, self).__init__()
        self.n = n
        self.k = k
        self.out_num = out_num
        self.temperature = temperature
        self.scores = torch.nn.Parameter(torch.randn(n, out_num))
        torch.nn.init.uniform_(self.scores, a=0, b=1)

    def forward(self, X):
        """
        Args:
            X: (B, n, D) Input features with batch size B, number of features per sample n, and dimension of each feature D.
        Returns:
            gates: (B, n, k) STE outputs for each channel (forward pass is discrete k-hot; backward pass gradients come from the soft distribution).
        """
        B, n, D, _, _,_ = X.shape
        assert n == self.n, "Input feature dimension n must match the initialized n"
        
        scores = self.scores.unsqueeze(0).expand(B, -1, -1)  # shape: (B, n, k)
        soft_scores = F.softmax(scores / self.temperature, dim=1)  # shape: (B, n, k)
        topk_indices = torch.topk(soft_scores, self.k, dim=1).indices  # shape: (B, k, k)
        
        # Create a zero tensor for the hard selections with shape (B, n, k)
        sparse_scores = torch.zeros_like(soft_scores)  # shape: (B, n, k)
        # Create batch indices with shape (B, k, out)
        batch_idx = torch.arange(B, device=X.device).view(B, 1, 1).expand(B, self.k, self.out_num)
        # Create channel indices with shape (B, k, out)
        channel_idx = torch.arange(self.out_num, device=X.device).view(1, 1, self.out_num).expand(B, self.k, self.out_num)
        sparse_scores[batch_idx, topk_indices, channel_idx] = soft_scores[batch_idx, topk_indices, channel_idx] 

        gates = sparse_scores - scores.detach() + scores  # shape: (B, n, k)
        gates = gates / gates.sum(dim=1, keepdim=True)
        return gates



class VitSplit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size,
        patch_size,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        out_indices=[2, 5, 8, 11],
        select_layers=[9, 10, 11],
        channels=384,

        output_orgimg=False,
        tuning_type="frozen",
        **vit_kwargs,
    ):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        backbone_model = ViT(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        proj_type="conv",
        pos_embed_type="learnable",
        classification=False,      # 作为 backbone 用，一般不需要 cls head
        spatial_dims=3,
        dropout_rate=0.0,
        qkv_bias=True,
        save_attn=False,
        window_size=0,
        window_block_indexes=(),
    )
        self.out_indices = tuple(out_indices)
        self.select_layers = tuple(select_layers)
        self.patch_size = patch_size  # (pD, pH, pW)
        self.select_layers=select_layers
        split_head=[]
        for layer_id in self.select_layers:
            copy_blk=copy.deepcopy(backbone_model.blocks[layer_id])
            split_head.append(copy_blk)
        self.split_head=nn.Sequential(*split_head)
        self.split_activations=None
        backbone_model.blocks[self.select_layers[0]-1].register_forward_hook(self.get_activation)

        backbone_model.register_forward_pre_hook(
            lambda _, x: CenterPadding(self.patch_size[0])(x[0])
        )

        if tuning_type == "frozen":
            for param in backbone_model.parameters():
                param.requires_grad = False

        elif tuning_type == "all":
            for param in backbone_model.parameters():
                param.requires_grad = True

        elif isinstance(tuning_type, list):
            for param in backbone_model.parameters():
                param.requires_grad = False

            for layer_id in tuning_type:
                for param in backbone_model.blocks[layer_id].parameters():
                    param.requires_grad = True
        else:
            raise AttributeError(f"{tuning_type} is not supported !!!!")
            
        self.backbone = backbone_model

        self.frozen_conv = nn.Sequential(*[
            nn.Conv3d(in_channels=channels*len(out_indices), out_channels=channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.GELU()
        ])
        self.fusion_conv = nn.Sequential(*[
            nn.Conv3d(in_channels=channels*2, out_channels=channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.GELU()
        ])
        self.bn_norm = nn.SyncBatchNorm(channels)

    @property
    def get_activation(self):
        def hook(model, input, output):
            self.split_activations = output.detach()
        return hook
    
    def reshape_vit_tokens(self, x, norm=True):
        """
        reshape vit tokens from (b, L, D) to (b, D, d, h, w)
        input:  x = (B, L, D)
        output: x_ = (B, D, d, h, w)
        """
        B, L, D = x.shape

        # optional LayerNorm from backbone
        if norm:
            x = self.backbone.norm(x)

        if hasattr(self.backbone, "cls_token"):
            x_ = x[:, 1:, :]
        else:
            x_ = x
        # now x_ should have exactly d*h*w tokens
        expected_len = self.d * self.h * self.w
        assert x_.shape[1] == expected_len, \
            f"Token count mismatch: expected {expected_len}, got {x_.shape[1]}"

        # reshape into 3D feature map
        # (B, d*h*w, D) → (B, d, h, w, D) → (B, D, d, h, w)
        x_ = (
            x_
            .reshape(B, self.d, self.h, self.w, D)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

        return x_
    def get_backbone_features(self, x):
        # ViT.forward: return x, hidden_states_out
        x, hidden_states_out = self.backbone(x)   # list of (B, L, D)
        # 只取 out_indices 中的层
        feats = [hidden_states_out[i] for i in self.out_indices]  # list len = n
        return x,feats,hidden_states_out
    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 1) 计算 token grid 尺寸
        pD, pH, pW = self.patch_size
        self.d = math.ceil(D / pD)
        self.h = math.ceil(H / pH)
        self.w = math.ceil(W / pW)

        # 2) Backbone 中间层 (list of (B, L, D))
        _,frozen_features_tokens,hidden_states_out = self.get_backbone_features(x)

        # 3) 将每层 tokens reshape 成 (B, C, d, h, w)，再在通道上拼起来
        frozen_maps = []
        for ft in frozen_features_tokens:
            fmap = self.reshape_vit_tokens(ft)       # (B, C, d, h, w)  这里假设 hidden_size == channels
            frozen_maps.append(fmap)

        frozen_features = torch.cat(frozen_maps, dim=1)  # (B, C * len(out_indices), d, h, w)
        frozen_features = self.frozen_conv(frozen_features)

        # 4) split_head branch
        tuned_tokens = self.split_head(self.split_activations)  # (B, L, D)
        tuned_features = self.reshape_vit_tokens(tuned_tokens)  # (B, C, d, h, w)

        # 5) 融合
        x = torch.cat([frozen_features, tuned_features], dim=1)  # (B, 2C, d, h, w)
        x = self.fusion_conv(x)                                  # (B, C, d, h, w)
        x=x.flatten(2)  
        return x,hidden_states_out