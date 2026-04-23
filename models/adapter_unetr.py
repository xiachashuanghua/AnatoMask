# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from models.vit_adapter import ViTAdapter
from monai.utils import ensure_tuple_rep


class AdapterUNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.feature_size = feature_size
        self.vit = ViTAdapter(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            window_size=6,
            window_block_indexes=(0,1,3,4,6,7,9,10),
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        )
        self.encoder1 = UnetrBasicBlock(spatial_dims=3, in_channels=in_channels, out_channels=self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder2 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=self.feature_size,
                                        kernel_size=3,  stride=1, norm_name='instance', res_block=True)
        self.encoder3 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=2*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder4 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=4*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder5 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=8*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.decoder4 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*8, out_channels=self.feature_size*4,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.decoder3 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*4, out_channels=self.feature_size*2,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.decoder2 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*2, out_channels=self.feature_size,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.decoder1 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=self.feature_size,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        # self.out = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=num_classes)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        f1, f2, f3, f4 = self.vit(x_in)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(f1)
        enc2 = self.encoder3(f2)
        enc3 = self.encoder4(f3)
        enc4 = self.encoder5(f4)
        dec2 = self.decoder4(enc4, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)
