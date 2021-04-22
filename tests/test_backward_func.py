"""Test calling c++ backward func from Python"""
import math

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.nn.modules.utils import _single, _pair, _triple

from actnn.cpp_extension.backward_func import (cudnn_convolution_backward,
    cudnn_convolution_transpose_backward)


class conv1d_explicit_backward(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        stride = (1, stride)
        padding = (0, padding)
        dilation = (1, dilation)

        ctx.save_for_backward(input, weight, bias)
        ctx.other_args = (stride, padding, dilation, groups)

        input = input.unsqueeze(2)
        weight = weight.unsqueeze(2)
        out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return out.squeeze(2)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        input = input.unsqueeze(2)
        weight = weight.unsqueeze(2)
        grad_output = grad_output.unsqueeze(2)

        stride, padding, dilation, groups = ctx.other_args
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        grad_input, grad_weight = cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                False, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        grad_input = grad_input.squeeze(2)
        grad_weight = grad_weight.squeeze(2)

        return grad_input, grad_weight, grad_bias, None, None, None, None


class conv2d_explicit_backward(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.other_args = (stride, padding, dilation, groups)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.other_args
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        grad_input, grad_weight = cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                False, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None


class conv3d_explicit_backward(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.other_args = (stride, padding, dilation, groups)
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.other_args
        padding = _triple(padding)
        stride = _triple(stride)
        dilation = _triple(dilation)

        grad_input, grad_weight = cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                False, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3, 4])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None


class conv_transpose1d_explicit_backward(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
        stride = (1, stride)
        padding = (0, padding)
        dilation = (1, dilation)

        ctx.save_for_backward(input, weight, bias)
        ctx.other_args = (stride, padding, output_padding, dilation, groups)

        input = input.unsqueeze(2)
        weight = weight.unsqueeze(2)
        out = F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        return out.squeeze(2)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding, output_padding, dilation, groups = ctx.other_args

        input = input.unsqueeze(2)
        weight = weight.unsqueeze(2)
        grad_output = grad_output.unsqueeze(2)

        padding = _pair(padding)
        output_padding = _pair(output_padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        grad_input, grad_weight = cudnn_convolution_transpose_backward(
                input, grad_output, weight, padding, output_padding, stride, dilation, groups,
                False, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        grad_input = grad_input.squeeze(2)
        grad_weight = grad_weight.squeeze(2)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None



class conv_transpose2d_explicit_backward(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.other_args = (stride, padding, output_padding, dilation, groups)
        return F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding, output_padding, dilation, groups = ctx.other_args
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        grad_input, grad_weight = cudnn_convolution_transpose_backward(
                input, grad_output, weight, padding, output_padding, stride, dilation, groups,
                False, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class conv_transpose3d_explicit_backward(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.other_args = (stride, padding, output_padding, dilation, groups)
        return F.conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding, output_padding, dilation, groups = ctx.other_args
        padding = _triple(padding)
        output_padding = _triple(output_padding)
        stride = _triple(stride)
        dilation = _triple(dilation)

        grad_input, grad_weight = cudnn_convolution_transpose_backward(
                input, grad_output, weight, padding, output_padding, stride, dilation, groups,
                False, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3, 4])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def test_conv1d_correctness():
    # arguments and test data
    N, H, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 28, 64, 128, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, H).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size).astype('float32')
    bias_np = np.random.rand(CO).astype('float32')


    def test_implementation(func):
        data = torch.tensor(data_np).to('cuda').requires_grad_()
        weight = torch.tensor(weight_np).to('cuda').requires_grad_()
        bias = torch.tensor(bias_np).to('cuda').requires_grad_()

        output = func(data, weight, bias, stride, padding, dilation, groups)
        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.conv1d)
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(conv1d_explicit_backward.apply)

    atol = 1e-5
    print("========== Conv1d Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol)


def test_conv2d_correctness():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 28, 28, 64, 128, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.rand(CO).astype('float32')


    def test_implementation(func):
        data = torch.tensor(data_np).to('cuda').requires_grad_()
        weight = torch.tensor(weight_np).to('cuda').requires_grad_()
        bias = torch.tensor(bias_np).to('cuda').requires_grad_()

        output = func(data, weight, bias, stride, padding, dilation, groups)
        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.conv2d)
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(conv2d_explicit_backward.apply)

    atol = 1e-5
    print("========== Conv2d Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol)


def test_conv3d_correctness():
    # arguments and test data
    N, D, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 16, 28, 28, 64, 128, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, D, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.rand(CO).astype('float32')

    def test_implementation(func):
        data = torch.tensor(data_np).to('cuda').requires_grad_()
        weight = torch.tensor(weight_np).to('cuda').requires_grad_()
        bias = torch.tensor(bias_np).to('cuda').requires_grad_()

        output = func(data, weight, bias, stride, padding, dilation, groups)
        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.conv3d)
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(conv3d_explicit_backward.apply)

    atol = 5e-4
    print("========== Conv3d Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol)


def test_conv1d_transpose_correctness():
    # arguments and test data
    N, H, CI, CO, kernel_size, stride, padding, output_padding, dilation, groups =\
	    4, 28, 64, 128, 3, 1, 1, 0, 1, 1
    data_np = np.random.randn(N, CI, H).astype('float32')
    weight_np = np.random.randn(CI, CO // groups, kernel_size).astype('float32')
    bias_np = np.random.rand(CO).astype('float32')


    def test_implementation(func):
        data = torch.tensor(data_np).to('cuda').requires_grad_()
        weight = torch.tensor(weight_np).to('cuda').requires_grad_()
        bias = torch.tensor(bias_np).to('cuda').requires_grad_()

        output = func(data, weight, bias, stride, padding, output_padding, dilation, groups)
        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.conv_transpose1d)
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(conv_transpose1d_explicit_backward.apply)

    atol = 2e-4
    print("========== Conv1dTranspose Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol)


def test_conv2d_transpose_correctness():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, output_padding, dilation, groups =\
	    4, 28, 28, 64, 128, 3, 1, 1, 0, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CI, CO // groups, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.rand(CO).astype('float32')


    def test_implementation(func):
        data = torch.tensor(data_np).to('cuda').requires_grad_()
        weight = torch.tensor(weight_np).to('cuda').requires_grad_()
        bias = torch.tensor(bias_np).to('cuda').requires_grad_()

        output = func(data, weight, bias, stride, padding, output_padding, dilation, groups)
        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.conv_transpose2d)
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(conv_transpose2d_explicit_backward.apply)

    atol = 2e-4
    print("========== Conv2dTranspose Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol)


def test_conv3d_transpose_correctness():
    # arguments and test data
    N, D, H, W, CI, CO, kernel_size, stride, padding, output_padding, dilation, groups =\
	    4, 16, 28, 28, 64, 128, 3, 1, 1, 0, 1, 1
    data_np = np.random.randn(N, CI, D, H, W).astype('float32')
    weight_np = np.random.randn(CI, CO // groups, kernel_size, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.rand(CO).astype('float32')


    def test_implementation(func):
        data = torch.tensor(data_np).to('cuda').requires_grad_()
        weight = torch.tensor(weight_np).to('cuda').requires_grad_()
        bias = torch.tensor(bias_np).to('cuda').requires_grad_()

        output = func(data, weight, bias, stride, padding, output_padding, dilation, groups)
        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.conv_transpose3d)
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(conv_transpose3d_explicit_backward.apply)

    atol = 2e-4
    print("========== Conv3dTranspose Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol)


if __name__ == "__main__":
    test_conv1d_correctness()
    test_conv2d_correctness()
    test_conv3d_correctness()

    test_conv1d_transpose_correctness()
    test_conv2d_transpose_correctness()
    test_conv3d_transpose_correctness()

