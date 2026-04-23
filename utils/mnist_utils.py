# import torch.nn as nn
# import numpy as np
# from .batchnorm import SynchronizedBatchNorm3d, SynchronizedBatchNorm2d

# class Transform3D:

#     def __init__(self, mul=None):
#         self.mul = mul

#     def __call__(self, voxel):
   
#         if self.mul == '0.5':
#             voxel = voxel * 0.5
#         elif self.mul == 'random':
#             voxel = voxel * np.random.uniform()
        
#         return voxel.astype(np.float32)


# def model_to_syncbn(model):
#     preserve_state_dict = model.state_dict()
#     _convert_module_from_bn_to_syncbn(model)
#     model.load_state_dict(preserve_state_dict)
#     return model


# def _convert_module_from_bn_to_syncbn(module):
#     for child_name, child in module.named_children(): 
#         if hasattr(nn, child.__class__.__name__) and \
#             'batchnorm' in child.__class__.__name__.lower():
#             TargetClass = globals()['Synchronized'+child.__class__.__name__]
#             arguments = TargetClass.__init__.__code__.co_varnames[1:]
#             kwargs = {k: getattr(child, k) for k in arguments}
#             setattr(module, child_name, TargetClass(**kwargs))
#         else:
#             _convert_module_from_bn_to_syncbn(child)
import torch.nn as nn
import numpy as np

try:
    from .batchnorm import SynchronizedBatchNorm3d, SynchronizedBatchNorm2d
except ImportError:
    SynchronizedBatchNorm3d = nn.BatchNorm3d
    SynchronizedBatchNorm2d = nn.BatchNorm2d

class Transform3D:

    def __init__(
        self,
        train=False,
        eval_scale=1.0,
        flip_prob=0.5,
        rotate_prob=0.5,
        intensity_scale=0.15,
        intensity_shift=0.10,
        noise_std=0.03,
        gamma_range=(0.9, 1.1),
        cutout_prob=0.3,
        cutout_ratio=0.2,
    ):
        self.train = train
        self.eval_scale = eval_scale
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.intensity_scale = intensity_scale
        self.intensity_shift = intensity_shift
        self.noise_std = noise_std
        self.gamma_range = gamma_range
        self.cutout_prob = cutout_prob
        self.cutout_ratio = cutout_ratio

    def _spatial_axes(self, voxel):
        if voxel.ndim == 3:
            return (0, 1, 2)
        if voxel.ndim == 4:
            if voxel.shape[0] in (1, 3) and voxel.shape[-1] not in (1, 3):
                return (1, 2, 3)
            return (0, 1, 2)
        raise ValueError(f"Unsupported voxel shape: {voxel.shape}")

    def _clip(self, voxel, orig_min, orig_max):
        if orig_min >= 0.0:
            upper = 1.0 if orig_max <= 1.5 else 255.0
            voxel = np.clip(voxel, 0.0, upper)
        return voxel

    def __call__(self, voxel):
        voxel = voxel.astype(np.float32, copy=True)
        orig_min = float(voxel.min())
        orig_max = float(voxel.max())

        if not self.train:
            voxel = voxel * self.eval_scale
            return voxel.astype(np.float32)

        spatial_axes = self._spatial_axes(voxel)

        for axis in spatial_axes:
            if np.random.rand() < self.flip_prob:
                voxel = np.flip(voxel, axis=axis).copy()

        if np.random.rand() < self.rotate_prob:
            plane_pairs = [
                (spatial_axes[0], spatial_axes[1]),
                (spatial_axes[0], spatial_axes[2]),
                (spatial_axes[1], spatial_axes[2]),
            ]
            plane = plane_pairs[np.random.randint(len(plane_pairs))]
            k = np.random.randint(0, 4)
            voxel = np.rot90(voxel, k=k, axes=plane).copy()

        value_range = max(orig_max - orig_min, 1e-6)
        scale = np.random.uniform(1.0 - self.intensity_scale, 1.0 + self.intensity_scale)
        shift = np.random.uniform(-self.intensity_shift, self.intensity_shift) * value_range
        voxel = voxel * scale + shift

        if self.gamma_range is not None and np.random.rand() < 0.5:
            gamma = np.random.uniform(*self.gamma_range)
            v_min = float(voxel.min())
            v_max = float(voxel.max())
            if v_max > v_min:
                voxel = (voxel - v_min) / (v_max - v_min + 1e-8)
                voxel = np.power(voxel, gamma)
                voxel = voxel * (v_max - v_min) + v_min

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std * value_range, size=voxel.shape)
            voxel = voxel + noise.astype(np.float32)

        if np.random.rand() < self.cutout_prob:
            slices = [slice(None)] * voxel.ndim
            for axis in spatial_axes:
                axis_len = voxel.shape[axis]
                cut_len = max(1, int(axis_len * self.cutout_ratio))
                start = np.random.randint(0, max(1, axis_len - cut_len + 1))
                slices[axis] = slice(start, start + cut_len)
            voxel[tuple(slices)] = 0.0

        voxel = self._clip(voxel, orig_min, orig_max)
        return voxel.astype(np.float32)


def model_to_syncbn(model):
    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model


def _convert_module_from_bn_to_syncbn(module):
    for child_name, child in module.named_children(): 
        if hasattr(nn, child.__class__.__name__) and \
            'batchnorm' in child.__class__.__name__.lower():
            TargetClass = globals()['Synchronized'+child.__class__.__name__]
            arguments = TargetClass.__init__.__code__.co_varnames[1:]
            kwargs = {k: getattr(child, k) for k in arguments}
            setattr(module, child_name, TargetClass(**kwargs))
        else:
            _convert_module_from_bn_to_syncbn(child)
