from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import MLPBlock, SABlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.layers import trunc_normal_
from monai.utils import ensure_tuple_rep

__all__ = [
    "MAEViTForClassification",
    "create_mae_vit_classifier",
    "load_pretrained_encoder",
]


class TransformerBlock(nn.Module):
    """Same encoder block layout as model_new.py, so pretrained keys match."""

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        with_cross_attention: bool = False,
        use_flash_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = True,
    ) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(
            hidden_size,
            num_heads,
            dropout_rate,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            causal=causal,
            sequence_length=sequence_length,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.with_cross_attention = with_cross_attention

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MAEViTForClassification(nn.Module):
    """
    Downstream 3D classification model for checkpoints produced by model_new.py.

    It keeps only the MAE encoder:
      patch_embedding + cls_token + register_tokens + transformer blocks

    The MAE decoders, mask tokens, teacher, reconstruction heads, and
    pretraining losses are intentionally removed for classification.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        num_classes: int | None = None,
        out_channels: int | None = None,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "sincos",
        dropout_rate: float = 0.2,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        num_register_tokens: int = 4,
        pooling: str = "cls",
        cls_dropout: float = 0.0,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if num_classes is None and out_channels is None:
            raise ValueError("Either num_classes or out_channels must be provided.")
        if num_classes is not None and out_channels is not None and num_classes != out_channels:
            raise ValueError("num_classes and out_channels must be equal when both are provided.")
        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate should be between 0 and 1, got {dropout_rate}.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_classes = int(num_classes if num_classes is not None else out_channels)
        self.hidden_size = hidden_size
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.spatial_dims = spatial_dims
        self.pooling = pooling

        for image_dim, patch_dim in zip(self.img_size, self.patch_size):
            if image_dim % patch_dim != 0:
                raise ValueError(f"patch_size={patch_size} should divide img_size={img_size}.")

        self.patch_grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.total_patches = int(np.prod(self.patch_grid_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.num_register_tokens = int(num_register_tokens)
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, hidden_size))
        else:
            self.register_tokens = None

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )

        blocks = [
            TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
            for _ in range(num_layers)
        ]
        self.blocks = nn.Sequential(*blocks, nn.LayerNorm(hidden_size))

        if pooling == "cls":
            head_dim = hidden_size
        elif pooling == "mean":
            head_dim = hidden_size
        elif pooling == "cls_mean":
            head_dim = hidden_size * 2
        else:
            raise ValueError("pooling must be one of: 'cls', 'mean', 'cls_mean'.")

        self.head_drop = nn.Dropout(cls_dropout) if cls_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(head_dim, self.num_classes)

        self._init_weights()
        if freeze_encoder:
            self.freeze_encoder()

    def _init_weights(self) -> None:
        trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-2.0, b=2.0)
        if self.register_tokens is not None:
            trunc_normal_(self.register_tokens, mean=0.0, std=0.02, a=-2.0, b=2.0)
        trunc_normal_(self.classifier.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def _add_special_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = patch_tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        if self.register_tokens is None:
            return torch.cat([cls, patch_tokens], dim=1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat([cls, register_tokens, patch_tokens], dim=1)

    def forward_features(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        patch_tokens = self.patch_embedding(x)
        encoded = self.blocks(self._add_special_tokens(patch_tokens))

        cls_feature = encoded[:, 0]
        patch_features = encoded[:, -self.total_patches :, :]
        mean_feature = patch_features.mean(dim=1)

        if self.pooling == "cls":
            feature = cls_feature
        elif self.pooling == "mean":
            feature = mean_feature
        else:
            feature = torch.cat([cls_feature, mean_feature], dim=-1)

        if return_tokens:
            return {
                "feature": feature,
                "cls_feature": cls_feature,
                "patch_features": patch_features,
                "tokens": encoded,
            }
        return feature

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feature = self.forward_features(x)
        logits = self.classifier(self.head_drop(feature))
        if return_features:
            return logits, feature
        return logits

    def freeze_encoder(self) -> None:
        for module in (self.patch_embedding, self.blocks):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.cls_token.requires_grad_(False)
        if self.register_tokens is not None:
            self.register_tokens.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def load_pretrained(
        self,
        checkpoint_path: str,
        strict: bool = False,
        use_teacher: bool = False,
        pretrain_img_size: Sequence[int] | int | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        return load_pretrained_encoder(
            self,
            checkpoint_path=checkpoint_path,
            strict=strict,
            use_teacher=use_teacher,
            pretrain_img_size=pretrain_img_size,
            verbose=verbose,
        )


def create_mae_vit_classifier(
    model_type: str = "vit-b",
    img_size: Sequence[int] | int = (96, 96, 96),
    patch_size: Sequence[int] | int = (16, 16, 16),
    in_channels: int = 1,
    num_classes: int | None = None,
    out_channels: int | None = None,
    spatial_dims: int = 3,
    num_register_tokens: int = 4,
    pooling: str = "cls",
    cls_dropout: float = 0.0,
    freeze_encoder: bool = False,
    save_attn: bool = False,
) -> MAEViTForClassification:
    model_configs = {
        "vit-t": {"hidden_size": 192, "mlp_dim": 768, "num_layers": 12, "num_heads": 3},
        "vit-b": {"hidden_size": 768, "mlp_dim": 3072, "num_layers": 12, "num_heads": 12},
        "vit-l": {"hidden_size": 1024, "mlp_dim": 4096, "num_layers": 24, "num_heads": 16},
    }
    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")

    config = model_configs[model_type]
    return MAEViTForClassification(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        out_channels=out_channels,
        hidden_size=config["hidden_size"],
        mlp_dim=config["mlp_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        spatial_dims=spatial_dims,
        num_register_tokens=num_register_tokens,
        pooling=pooling,
        cls_dropout=cls_dropout,
        freeze_encoder=freeze_encoder,
        save_attn=save_attn,
    )


def load_pretrained_encoder(
    model: MAEViTForClassification,
    checkpoint_path: str,
    strict: bool = False,
    use_teacher: bool = False,
    pretrain_img_size: Sequence[int] | int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load only encoder-compatible weights from a model_new.py checkpoint.

    Supported checkpoint formats:
      {"state_dict": ...}, {"model": ...}, {"net": ...}, or a raw state_dict.

    Decoder, teacher, projector, reconstruction, and segmentation weights are
    ignored. The classifier is always initialized from scratch.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    adapted_state: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}

    for raw_key, value in state_dict.items():
        key = _normalize_pretrained_key(raw_key, use_teacher=use_teacher)
        if key is None:
            skipped[raw_key] = "unused_pretraining_component"
            continue
        if key.startswith("classifier.") or key.startswith("head_drop."):
            skipped[raw_key] = "classification_head"
            continue
        if key not in model_state:
            skipped[raw_key] = "not_in_classification_model"
            continue

        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            adapted_state[key] = value
            continue

        if key == "patch_embedding.position_embeddings":
            resized = _resize_position_embedding(
                value,
                target,
                model.patch_grid_size,
                model.patch_size,
                pretrain_img_size=pretrain_img_size,
            )
            if resized is not None:
                adapted_state[key] = resized
                continue

        if key.endswith("patch_embeddings.weight"):
            adapted = _adapt_input_conv_weight(value, target)
            if adapted is not None:
                adapted_state[key] = adapted
                continue

        skipped[raw_key] = f"shape_mismatch:{tuple(value.shape)}->{tuple(target.shape)}"

    incompatible = model.load_state_dict(adapted_state, strict=False)

    report = {
        "loaded": sorted(adapted_state.keys()),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
        "skipped": skipped,
    }

    if strict:
        critical_missing = [
            key
            for key in report["missing"]
            if not key.startswith("classifier.") and not key.startswith("head_drop.")
        ]
        if critical_missing or report["unexpected"]:
            raise RuntimeError(
                "Strict pretrained loading failed. "
                f"critical_missing={critical_missing}, unexpected={report['unexpected']}"
            )

    if verbose:
        print(
            "Loaded pretrained encoder weights: "
            f"{len(report['loaded'])} tensors loaded, "
            f"{len(report['missing'])} missing, "
            f"{len(report['skipped'])} skipped."
        )
        if report["missing"]:
            print("Missing keys include classifier weights, which is expected for downstream classification.")

    return report


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("checkpoint must be a state_dict or a dict containing state_dict/model/net.")


def _normalize_pretrained_key(key: str, use_teacher: bool = False) -> str | None:
    prefixes_to_strip = ("module.", "model.", "backbone.", "encoder.")
    for prefix in prefixes_to_strip:
        if key.startswith(prefix):
            key = key[len(prefix) :]

    if key.startswith("vit."):
        key = key[len("vit.") :]

    if key.startswith("teacher."):
        if not use_teacher:
            return None
        key = key[len("teacher.") :]
        if key.startswith("proj_head."):
            return None

    unused_prefixes = (
        "decoder",
        "decoder2",
        "mask_tokens",
        "mask_tokens1",
        "mask_tokens2",
        "neigh_attn",
        "patch_proj",
        "feat_proj",
        "proj_head_student",
        "voxel_seg_head1",
        "tube_neighbor_idx",
        "tube_neighbor_valid",
    )
    if key.startswith(unused_prefixes):
        return None
    return key


def _resize_position_embedding(
    source: torch.Tensor,
    target: torch.Tensor,
    current_grid: Sequence[int],
    patch_size: Sequence[int],
    pretrain_img_size: Sequence[int] | int | None = None,
) -> torch.Tensor | None:
    if source.ndim != 3 or target.ndim != 3:
        return None
    if source.shape[0] != target.shape[0] or source.shape[-1] != target.shape[-1]:
        return None

    old_tokens = source.shape[1]
    new_tokens = target.shape[1]
    if old_tokens == new_tokens:
        return source

    old_grid = _infer_old_grid(old_tokens, patch_size, pretrain_img_size)
    if old_grid is None or int(np.prod(old_grid)) != old_tokens:
        return None
    if int(np.prod(current_grid)) != new_tokens:
        return None

    embed_dim = source.shape[-1]
    source_3d = source.reshape(1, old_grid[0], old_grid[1], old_grid[2], embed_dim)
    source_3d = source_3d.permute(0, 4, 1, 2, 3).contiguous()
    resized = F.interpolate(source_3d, size=tuple(current_grid), mode="trilinear", align_corners=False)
    resized = resized.permute(0, 2, 3, 4, 1).reshape(1, new_tokens, embed_dim)
    return resized.to(dtype=target.dtype)


def _infer_old_grid(
    old_tokens: int,
    patch_size: Sequence[int],
    pretrain_img_size: Sequence[int] | int | None = None,
) -> tuple[int, int, int] | None:
    if pretrain_img_size is not None:
        pretrain_img_size = ensure_tuple_rep(pretrain_img_size, 3)
        patch_size = ensure_tuple_rep(patch_size, 3)
        return tuple(int(s // p) for s, p in zip(pretrain_img_size, patch_size))

    side = round(old_tokens ** (1.0 / 3.0))
    if side ** 3 == old_tokens:
        return (side, side, side)
    return None


def _adapt_input_conv_weight(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if source.ndim != target.ndim or source.ndim not in (4, 5):
        return None
    if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
        return None

    source_channels = source.shape[1]
    target_channels = target.shape[1]
    if source_channels == target_channels:
        return source
    if source_channels == 1 and target_channels > 1:
        return source.repeat(1, target_channels, *([1] * (source.ndim - 2))) / target_channels
    if target_channels == 1 and source_channels > 1:
        return source.mean(dim=1, keepdim=True)
    return None
