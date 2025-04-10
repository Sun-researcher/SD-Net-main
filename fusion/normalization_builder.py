from torch import nn
from warnings import warn


def buildNormalization(norm_name,
                       feature_shape,
                       affine=True,
                       norm_name_map=None,
                       **kwargs):
    # 检查是否指定了norm名称的映射
    if norm_name_map is not None:
        if norm_name in norm_name_map:
            norm_name = norm_name_map[norm_name]

    if norm_name not in norm_switch:
        warn(f'[buildNormalization] Unrecognized norm type: {norm_name}. Return identity instead.')

    return norm_switch.get(norm_name, _identity)(feature_shape, affine, **kwargs)


def _batchNorm1D(dim, affine, **kwargs):
    return nn.BatchNorm1d(dim, affine=affine)


def _batchNorm2D(dim, affine, **kwargs):
    return nn.BatchNorm2d(dim, affine=affine)


def _layerNorm(feature_shape, affine, **kwargs):
    return nn.LayerNorm(feature_shape, elementwise_affine=affine)


def _instanceNorm1D(dim, affine, **kwargs):
    return nn.InstanceNorm1d(dim, affine=affine)


def _identity(dim, affine, **kwargs):
    return nn.Identity()


norm_switch = {
    'bn_1d': _batchNorm1D,
    'bn_2d': _batchNorm2D,
    'ln': _layerNorm,
    'in_1d': _instanceNorm1D,

    None: _identity,
    'none': _identity,
    'identity': _identity
}