# The code is compatible with PyTorch 1.6/1.7
from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from torch import Tensor
from torch.nn.modules.pooling import _size_2_t, _single, _pair, _triple, _MaxPoolNd, _AvgPoolNd

from actnn.qscheme import QScheme
from actnn.qbnscheme import QBNScheme
from actnn.conf import config
from actnn.ops import linear, batch_norm, conv1d, conv2d, conv3d, sync_batch_norm
from actnn.ops import conv_transpose1d, conv_transpose2d, conv_transpose3d
import actnn.cpp_extension.quantization as ext_quantization


class QConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', group=0):
        super(QConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode)
        if isinstance(kernel_size, int):
            num_locations = kernel_size
        else:
            num_locations = kernel_size[0]

        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, num_locations=num_locations, group=group, depthwise_groups=groups)
        else:
            self.scheme = None

    def forward(self, input):
        if config.training:
            if self.padding_mode != 'zeros':
                return conv1d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                    self.weight, self.bias, self.stride,
                                    _single(0), self.dilation, self.groups, self.scheme)
            return conv1d.apply(input, self.weight, self.bias, self.stride,
                                 self.padding, self.dilation, self.groups, self.scheme)
        else:
            return super(QConv1d, self).forward(input)


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', group=0):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode)
        if isinstance(kernel_size, int):
            num_locations = kernel_size ** 2
        else:
            num_locations = kernel_size[0] * kernel_size[1]

        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, num_locations=num_locations, group=group, depthwise_groups=groups)
        else:
            self.scheme = None

    def forward(self, input):
        if config.training:
            if self.padding_mode != 'zeros':
                return conv2d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                    self.weight, self.bias, self.stride,
                                    _pair(0), self.dilation, self.groups, self.scheme)
            return conv2d.apply(input, self.weight, self.bias, self.stride,
                                 self.padding, self.dilation, self.groups, self.scheme)
        else:
            return super(QConv2d, self).forward(input)
        

class QConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', group=0):
        super(QConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode)
        if isinstance(kernel_size, int):
            num_locations = kernel_size ** 3
        else:
            num_locations = kernel_size[0] * kernel_size[1] * kernel_size[2]

        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, num_locations=num_locations, group=group, depthwise_groups=groups)
        else:
            self.scheme = None

    def forward(self, input):
        if config.training:
            if self.padding_mode != 'zeros':
                return conv3d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                    self.weight, self.bias, self.stride,
                                    _triple(0), self.dilation, self.groups, self.scheme)
            return conv3d.apply(input, self.weight, self.bias, self.stride,
                                 self.padding, self.dilation, self.groups, self.scheme)
        else:
            return super(QConv3d, self).forward(input)


class QConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', group=0):
        super(QConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, output_padding, groups, bias, dilation, padding_mode)
        if isinstance(kernel_size, int):
            num_locations = kernel_size
        else:
            num_locations = kernel_size[0]

        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, num_locations=num_locations, group=group, depthwise_groups=groups)
        else:
            self.scheme = None

    def forward(self, input, output_size=None):
        if config.training:
            if self.padding_mode != 'zeros':
                raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore

            return conv_transpose1d.apply(
                input, self.weight, self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation, self.scheme)
        else:
            return super(QConvTranspose1d, self).forward(input, output_size)


class QConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', group=0):
        super(QConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, output_padding, groups, bias, dilation, padding_mode)
        if isinstance(kernel_size, int):
            num_locations = kernel_size ** 2
        else:
            num_locations = kernel_size[0] * kernel_size[1]

        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, num_locations=num_locations, group=group, depthwise_groups=groups)
        else:
            self.scheme = None

    def forward(self, input, output_size=None):
        if config.training:
            if self.padding_mode != 'zeros':
                raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore

            return conv_transpose2d.apply(
                input, self.weight, self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation, self.scheme)
        else:
            return super(QConvTranspose2d, self).forward(input, output_size)


class QConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', group=0):
        super(QConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, output_padding, groups, bias, dilation, padding_mode)
        if isinstance(kernel_size, int):
            num_locations = kernel_size ** 3
        else:
            num_locations = kernel_size[0] * kernel_size[1] * kernel_size[2]

        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, num_locations=num_locations, group=group, depthwise_groups=groups)
        else:
            self.scheme = None

    def forward(self, input, output_size=None):
        if config.training:
            if self.padding_mode != 'zeros':
                raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore

            return conv_transpose3d.apply(
                input, self.weight, self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation, self.scheme)
        else:
            return super(QConvTranspose3d, self).forward(input, output_size)


class QLinear(nn.Linear):
    num_layers = 0

    def __init__(self, input_features, output_features, bias=True, group=0):
        super(QLinear, self).__init__(input_features, output_features, bias)
        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, group=group)
        else:
            self.scheme = None

    def forward(self, input):
        if config.training:
            return linear.apply(input, self.weight, self.bias, self.scheme)
        else:
            return super(QLinear, self).forward(input)


class QBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(QBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if config.adaptive_bn_scheme:
            self.scheme = QBNScheme(group=group)
        else:
            self.scheme = None

    def forward(self, input):
        if not config.training:
            return super(QBatchNorm1d, self).forward(input)

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return batch_norm.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps, self.scheme)


class QBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(QBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if config.adaptive_bn_scheme:
            self.scheme = QBNScheme(group=group)
        else:
            self.scheme = None

    def forward(self, input):
        if not config.training:
            return super(QBatchNorm2d, self).forward(input)

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return batch_norm.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps, self.scheme)


class QBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(QBatchNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if config.adaptive_bn_scheme:
            self.scheme = QBNScheme(group=group)
        else:
            self.scheme = None

    def forward(self, input):
        if not config.training:
            return super(QBatchNorm3d, self).forward(input)

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return batch_norm.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps, self.scheme)


class QReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ext_quantization.act_quantized_relu(input)


class QDropout(nn.Dropout):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return ext_quantization.act_quantized_dropout(input, self.p)
        else:
            return super(QDropout, self).forward(input)


class QSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group=None,
        group=0
    ) -> None:
        super(QSyncBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats, process_group)
        if config.adaptive_bn_scheme:
            self.scheme = QBNScheme(group=group)
        else:
            self.scheme = None

    def forward(self, input):
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None

        need_sync = bn_training
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return batch_norm().apply(
                input, running_mean, running_var, self.weight, self.bias,
                bn_training, exponential_average_factor, self.eps, self.scheme)
        else:
            if not self.ddp_gpu_size:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            assert bn_training
            return sync_batch_norm().apply(
                input, self.weight, self.bias, running_mean, running_var,
                self.eps, exponential_average_factor, process_group, world_size, self.scheme)


class QMaxPool2d(_MaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def forward(self, input):
        return ext_quantization.act_quantized_max_pool2d(
            input, self.kernel_size, self.stride,
            self.padding, self.dilation, self.ceil_mode,
            self.return_indices)


class QAvgPool2d(_AvgPoolNd):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: bool = None) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if (stride is not None) else kernel_size)
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        # TODO: implement memory-optimized cuda kernel for this.
        #return F.avg_pool2d(input, self.kernel_size, self.stride,
        #                    self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        warnings.warn("avg_pool2d is replcaed by max_pool2d, because the optimized cuda kernel"
                      "for avg_pool2d is not implemented.")
        return ext_quantization.act_quantized_max_pool2d(
            input, self.kernel_size, self.stride,
            self.padding, (1, 1), self.ceil_mode,
            False)

