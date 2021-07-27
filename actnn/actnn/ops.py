from collections import namedtuple
import os
import time

import numpy as np
import torch
from torch.autograd.function import Function
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.utils.cpp_extension import load

from actnn.conf import config
from actnn.utils import get_memory_usage, compute_tensor_bytes, empty_cache, swap_to_cpu
import actnn.cpp_extension.quantization as ext_quantization
import actnn.cpp_extension.minimax as ext_minimax
import actnn.cpp_extension.backward_func as ext_backward_func

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])


def quantize_and_pack(data, bits, mn, mx):
    if config.simulate:
        N = data.shape[0]
        output = data   # N, groups, group_dim

        if isinstance(bits, int):  # Handle the case when config.adaptive_scheme is False
            bits = torch.ones(N, dtype=torch.int32, device='cuda') * bits

        B = (2 ** bits - 1).view(N, 1, 1)
        mn = mn - 1e-6
        mx = mx + 1e-6
        scale = B / (mx - mn)     # N, groups, 1
        output = (output - mn) * scale

        if config.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output = F.relu(output)
        output = torch.min(output, B.float()).round_().int()
    else:
        # Pack to bitstream
        if isinstance(bits, int):
            pack_func = ext_quantization.pack_single_precision
        else:
            pack_func = ext_quantization.pack_mixed_precision
        output, scale = pack_func(data, mn, mx, bits, config.stochastic)
        if config.swap:
            output = swap_to_cpu(output)

    return output, scale


def dequantize_and_unpack(data, shape, bits, scale, mn):
    if config.simulate:
        data = data / scale + mn
    else:
        if config.swap:
            data = data.cuda(non_blocking=True)

        # Pad to group_size
        N = shape[0]
        num_features = int(np.prod(shape[1:]))
        group_size = config.group_size
        num_features = (num_features + (group_size - num_features % group_size) % group_size)

        # Unpack bitstream
        if isinstance(bits, int):
            unpack_func = ext_quantization.unpack_single_precision
        else:
            unpack_func = ext_quantization.unpack_mixed_precision
        data = unpack_func(data, bits, scale, mn, N, num_features // group_size, group_size)
    return data


def no_scheme_compute_quantization_bits(input):
    N = input.shape[0]
    D = input.shape[1]
    input_flatten = input.view(N, -1)
    num_features = input_flatten.shape[1]
    num_pixels = num_features // D

    # Compute min, max by groups
    if num_features % config.group_size != 0:
        # Padding
        new_num_features = (num_features // config.group_size + 1) * config.group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

    input_groups = input_flatten.view(-1, config.group_size)
    mn, mx = ext_minimax.minimax(input_groups)

    b = config.activation_compression_bits[0]
    return input_groups.view(N, -1, config.group_size), b, mn.view(N, -1, 1), mx.view(N, -1, 1)


def quantize_activation(input, scheme):
    if not config.compress_activation:
        if config.swap:
            input = swap_to_cpu(input)

        return input, None, None, None

    N = input.shape[0]
    if scheme:
        input_groups, q_bits, q_min, mx = scheme.compute_quantization_bits(input)
    else:
        input_groups, q_bits, q_min, mx = no_scheme_compute_quantization_bits(input)

    q_input, q_scale = quantize_and_pack(input_groups, q_bits, q_min, mx)

    # TODO convert q_bits to int8
    if input.dtype == torch.float32:
        return q_input, q_bits, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)
    else:
        return q_input, q_bits, q_scale, q_min


def dequantize_activation(quantized, q_input_shape):
    if not config.compress_activation:
        ret = quantized[0]
        if config.swap:
            ret = ret.cuda(non_blocking=True)
        return ret

    q_input, q_bits, q_scale, q_min = quantized
    if q_scale.dtype == torch.bfloat16:
        q_scale = q_scale.to(torch.float32)
        q_min = q_min.to(torch.float32)
    input = dequantize_and_unpack(q_input, q_input_shape, q_bits, q_scale, q_min)

    # Remove padding
    N = q_input_shape[0]
    num_features = np.prod(q_input_shape[1:])
    input = input.view(N, -1)[:, :num_features]
    input = input.view(*q_input_shape)
    return input.contiguous()

conv2d_layer_ct = 0
bn_layer_ct = 0
total_act_mem = 0

class convnd(Function):
    @staticmethod
    def run_forward(n, forward_op, ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        quantized = quantize_activation(input, scheme)

        ctx.scheme = scheme
        ctx.saved = quantized, weight, bias
        ctx.other_args = (input.shape, stride, padding, dilation, groups)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global conv2d_layer_ct, total_act_mem
            print("========== conv%dd forward %d ==========" % (d, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            total_act_mem += compute_tensor_bytes(quantized)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        return forward_op(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def run_backward(n, ctx, grad_output, bias_reduce_dims, aug):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return None, None, None, None, None, None, None, None
        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)

        q_input_shape, stride, padding, dilation, groups = ctx.other_args
        padding = aug(padding)
        stride = aug(stride)
        dilation = aug(dilation)

        quantized, weight, bias = ctx.saved
        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global conv2d_layer_ct
            print("========== conv%dd backward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            print("WS: %.2f MB" % (compute_tensor_bytes([grad_output, input, input]) / 1024 ** 2))

        use_pipeline = False
        if config.pipeline_threshold:
            ws_mem = compute_tensor_bytes([grad_output, input, input])
            if (ws_mem > config.pipeline_threshold and
                ctx.needs_input_grad[1] and ctx.needs_input_grad[0]):
                use_pipeline = True

        if use_pipeline:
            micro_batch_size = (ws_mem + config.pipeline_threshold) // config.pipeline_threshold
            raw_input = input
            raw_grad_output = grad_output
            input = torch.chunk(input, micro_batch_size)
            grad_output = torch.chunk(grad_output,  micro_batch_size)
            grad_weight = None

            for i in range(micro_batch_size):
                input[i][:], grad_weight_tmp = ext_backward_func.cudnn_convolution_backward(
                        input[i], grad_output[i], weight, padding, stride, dilation, groups,
                        config.cudnn_benchmark_conv2d, False, False,
                        [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])
                if grad_weight is None:
                    grad_weight = grad_weight_tmp
                else:
                    grad_weight += grad_weight_tmp
            grad_input = raw_input
            grad_output = raw_grad_output
        else:
            grad_input, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                config.cudnn_benchmark_conv2d, False, False,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(bias_reduce_dims)
        else:
            grad_bias = None

        if ctx.scheme:
            ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class conv1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
        return convnd.run_forward(1, F.conv1d, ctx, input, weight, bias, stride, padding, dilation, groups, scheme)

    @staticmethod
    def backward(ctx, grad_output):
        return convnd.run_backward(1, ctx, grad_output, [0, 2], _single)


class conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
        return convnd.run_forward(2, F.conv2d, ctx, input, weight, bias, stride, padding, dilation, groups, scheme)

    @staticmethod
    def backward(ctx, grad_output):
        return convnd.run_backward(2, ctx, grad_output, [0, 2, 3], _pair)


class conv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
        return convnd.run_forward(3, F.conv3d, ctx, input, weight, bias, stride, padding, dilation, groups, scheme)

    @staticmethod
    def backward(ctx, grad_output):
        return convnd.run_backward(3, ctx, grad_output, [0, 2, 3, 4], _triple)


class conv_transposend(Function):
    @staticmethod
    def run_forward(n, forward_op, ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, scheme=None):
        quantized = quantize_activation(input, scheme)

        ctx.scheme = scheme
        ctx.saved = quantized, weight, bias
        ctx.other_args = (input.shape, stride, padding, output_padding, dilation, groups)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global conv2d_layer_ct, total_act_mem
            print("========== conv%dd_transpose forward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            total_act_mem += compute_tensor_bytes(quantized)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        return forward_op(input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    def run_backward(n, ctx, grad_output, bias_reduce_dims, aug):
        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)

        q_input_shape, stride, padding, output_padding, dilation, groups = ctx.other_args
        padding = aug(padding)
        output_padding = aug(output_padding)
        stride = aug(stride)
        dilation = aug(dilation)

        quantized, weight, bias = ctx.saved
        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global conv2d_layer_ct
            print("========== conv%dd_transpose backward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            print("WS: %.2f MB" % (compute_tensor_bytes([grad_output, input, input]) / 1024 ** 2))

        use_pipeline = False
        if config.pipeline_threshold:
            ws_mem = compute_tensor_bytes([grad_output, input, input])
            if (ws_mem > config.pipeline_threshold and
                ctx.needs_input_grad[1] and ctx.needs_input_grad[0]):
                use_pipeline = True

        if use_pipeline:
            micro_batch_size = (ws_mem + config.pipeline_threshold) // config.pipeline_threshold
            raw_input = input
            raw_grad_output = grad_output
            input = torch.chunk(input, micro_batch_size)
            grad_output = torch.chunk(grad_output,  micro_batch_size)
            grad_weight = None

            for i in range(micro_batch_size):
                input[i][:], grad_weight_tmp = ext_backward_func.cudnn_convolution_transpose_backward(
                        input[i], grad_output[i], weight, padding, output_padding, stride, dilation, groups,
                        config.cudnn_benchmark_conv2d, False, False,
                        [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])
                if grad_weight is None:
                    grad_weight = grad_weight_tmp
                else:
                    grad_weight += grad_weight_tmp
            grad_input = raw_input
            grad_output = raw_grad_output
        else:
            grad_input, grad_weight = ext_backward_func.cudnn_convolution_transpose_backward(
                input, grad_output, weight, padding, output_padding, stride, dilation, groups,
                config.cudnn_benchmark_conv2d, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(bias_reduce_dims)
        else:
            grad_bias = None

        if ctx.scheme:
            ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class conv_transpose1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, scheme=None):
        return conv_transposend.run_forward(1, F.conv_transpose1d, ctx, input, weight, bias, stride,
                                            padding, output_padding, groups, dilation, scheme)

    @staticmethod
    def backward(ctx, grad_output):
        return conv_transposend.run_backward(1, ctx, grad_output, [0, 2], _single)


class conv_transpose2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, scheme=None):
        return conv_transposend.run_forward(2, F.conv_transpose2d, ctx, input, weight, bias, stride,
                                            padding, output_padding, groups, dilation, scheme)

    @staticmethod
    def backward(ctx, grad_output):
        return conv_transposend.run_backward(2, ctx, grad_output, [0, 2, 3], _pair)


class conv_transpose3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, scheme=None):
        return conv_transposend.run_forward(3, F.conv_transpose3d, ctx, input, weight, bias, stride,
                                            padding, output_padding, groups, dilation, scheme)

    @staticmethod
    def backward(ctx, grad_output):
        return conv_transposend.run_backward(3, ctx, grad_output, [0, 2, 3, 4], _triple)


class linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, scheme=None):
        quantized = quantize_activation(input, scheme)

        empty_cache(config.empty_cache_threshold)

        ctx.scheme = scheme
        ctx.saved = quantized, weight, bias
        ctx.other_args = input.shape

        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)

        quantized, weight, bias = ctx.saved
        q_input_shape = ctx.other_args

        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        # TODO: the following implementation might not be optimal
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]
        # rank = len(grad_output.shape)

        grad_output_flatten = grad_output.view(-1, C_out)
        input_flatten = input.view(-1, C_in)
        # print(grad_output_flatten.shape, weight.shape)
        grad_input = grad_output_flatten.mm(weight)
        grad_weight = grad_output_flatten.t().mm(input_flatten)

        # grad_input = grad_output.mm(weight)
        # grad_weight = grad_output.t().mm(input)
        if bias is not None:
            # grad_bias = grad_output.sum(0)
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None

        if ctx.scheme:
            ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None


class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps, scheme):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return ext_backward_func.cudnn_batch_norm(
        #         input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)[0]
        quantized = quantize_activation(input, scheme)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global bn_layer_ct, total_act_mem
            print("========== bn forward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1
            total_act_mem += compute_tensor_bytes(quantized)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        if training:
            output, save_mean, save_var, reserve = ext_backward_func.cudnn_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
        else:
            output, save_mean, save_var = ext_backward_func.native_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
            reserve = None

        ctx.scheme = scheme
        ctx.other_args = input.shape
        ctx.saved = (quantized, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return None, None, None, None, None, None, None, None, None
        quantized, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved

        q_input_shape = ctx.other_args

        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global bn_layer_ct
            print("========== bn backward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1

        if training:
            input = input.contiguous()
            grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
                input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        else:
            grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
                grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
            )

        if ctx.scheme:
            ctx.scheme.if_allocate_perlayer()
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None


class sync_batch_norm(Function):
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size, scheme):
        input = input.contiguous()

        count = torch.empty(1,
                            dtype=running_mean.dtype,
                            device=input.device).fill_(input.numel() // input.size(1))

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        num_channels = input.shape[1]
        # C, C, 1 -> (2C + 1)
        combined = torch.cat([mean, invstd, count], dim=0)
        # world_size * (2C + 1)
        combined_list = [
            torch.empty_like(combined) for k in range(world_size)
        ]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(combined_list, combined, process_group, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        size = count_all.view(-1).long().sum()
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )

        quantized = quantize_activation(input, scheme)
        self.saved = quantized
        self.save_for_backward(weight, mean, invstd, count_all)
        self.scheme = scheme
        self.other_args = input.shape
        self.process_group = process_group

        # apply element-wise normalization
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()

        quantized = self.saved
        q_input_shape = self.other_args
        saved_input = dequantize_activation(quantized, q_input_shape)
        del quantized, self.saved

        weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel
            num_channels = sum_dy.shape[0]
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
            torch.distributed.all_reduce(
                combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
            sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

            divisor = count_tensor.sum()
            mean_dy = sum_dy / divisor
            mean_dy_xmu = sum_dy_xmu / divisor
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        if self.scheme:
            self.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


class adaptive_avg_pool2d(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        assert output_size == (1, 1)
        ctx.saved = input.shape
        return torch.mean(input, dim=[2, 3], keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.saved
        repeat_size = [int(x / y) for x, y in zip(input_shape, grad_output.shape)]
        return grad_output.repeat(repeat_size) / np.prod(repeat_size), None

