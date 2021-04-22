"""Test the activation quantized ops"""

import math

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.autograd.function import Function

from timeit_v2 import py_benchmark

from actnn import QScheme, QBNScheme, config, get_memory_usage, compute_tensor_bytes
from actnn.ops import ext_backward_func, ext_quantization
from actnn.ops import conv2d as quantized_conv2d, batch_norm as quantized_batch_norm, \
        adaptive_avg_pool2d as quantized_adaptive_avg_pool2d


def test_relu_correctness():
    print("========== ReLU Correctness Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(128, 56, 56, 31).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            output = func(data)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref =  test_implementation(F.relu)
        output_us, grad_data_us = test_implementation(ext_quantization.act_quantized_relu)

        np.testing.assert_allclose(output_ref, output_us)
        np.testing.assert_allclose(grad_data_ref, grad_data_us)


def test_relu_memory():
    print("========== ReLU Memory Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(128, 56, 56, 32).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            before = get_memory_usage()

            for i in range(10):
                data = func(data)

            after = get_memory_usage()
            
            return after - before

        usage_ref = test_implementation(F.relu)
        usage_us = test_implementation(ext_quantization.act_quantized_relu)

        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))


def test_relu_speed():
    print("========== ReLU Speed Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")

        data_np = np.random.randn(256, 56, 56, 32).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            stmt = "func(data)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data)
            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.relu)
        forward_us, backward_us = test_implementation(ext_quantization.act_quantized_relu)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_adaptive_avg_pool2d_correctness():
    """Test the correctness of computation results"""
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 28, 28, 256, 256, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    head_np = np.random.randn(N, CI, 1, 1).astype('float32')
    output_size = 1, 1

    def test_implementation(func):
        torch.manual_seed(0)
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        head = torch.tensor(head_np).to("cuda")

        output = func(data, output_size)
        output.backward(head)

        return [x.detach().cpu().numpy() for x in [output, data.grad]]

    output_ref, grad_data_ref = test_implementation(F.adaptive_avg_pool2d)
    output_us, grad_data_us = test_implementation(quantized_adaptive_avg_pool2d.apply)

    atol = 1e-4
    rtol = 1e-4
    print("========== AdaptiveAvgPool2d Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol, rtol=rtol)


def test_adaptive_avg_pool2d_memory():
    """Test the memory usage"""
    # arguments and test data
    N, H, W, CI = 1024, 4, 4, 1024
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    output_size = (1, 1)

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = func(data, output_size)
        for i in range(10):
            output = func(output, output_size)

        return get_memory_usage() - compute_tensor_bytes([data, output])

    usage_ref = test_implementation(F.adaptive_avg_pool2d)
    usage_us = test_implementation(quantized_adaptive_avg_pool2d.apply)

    print("========== AdaptiveAvgPool2d Memory Test ==========")
    print("Exact.     Usage: %.3f MB" % (usage_ref / 2 ** 20))
    print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))


def test_max_pool2d_correctness():
    """Test the correctness of computation results"""
    # arguments and test data
    N, H, W, CI, kernel_size, stride, padding, dilation = 4, 28, 28, 8, 3, 2, 1, 1
    ceil_mode, return_indices = False, False

    print("========== MaxPool2d Correctness Test ==========")
    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(N, CI, H, W).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            output = func(data, (kernel_size, kernel_size), (stride, stride), (padding, padding),
                                (dilation, dilation), ceil_mode, return_indices)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref = test_implementation(F.max_pool2d)
        output_us, grad_data_us = test_implementation(ext_quantization.act_quantized_max_pool2d)

        atol = 1e-4
        rtol = 1e-4
        np.testing.assert_allclose(output_ref, output_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol, rtol=rtol)


def test_max_pool2d_memory():
    """Test the memory usage"""
    # arguments and test data
    N, H, W, CI, kernel_size, stride, padding, dilation = 128, 28, 28, 8, 3, 2, 1, 1
    ceil_mode, return_indices = False, False

    print("========== MaxPool2d Memory Test ==========")
    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")

        data_np = np.random.randn(N, CI, H, W).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            output = func(data, (kernel_size, kernel_size), (stride, stride), (padding, padding),
                              (dilation, dilation), ceil_mode, return_indices)

            return get_memory_usage() - compute_tensor_bytes([output, data])

        usage_ref = test_implementation(F.max_pool2d)
        usage_us = test_implementation(ext_quantization.act_quantized_max_pool2d)
        print("Exact.     Usage: %.3f MB" % (usage_ref / 2 ** 20))
        print("Quantized. Usage: %.3f MB" % (usage_us / 2 ** 20))


def test_max_pool2d_speed():
    """Test the correctness of computation results"""
    # arguments and test data
    N, H, W, CI, kernel_size, stride, padding, dilation = 128, 28, 28, 128, 3, 2, 1, 1
    ceil_mode, return_indices = False, False


    print("========== MaxPool2d Speed Test ==========")
    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(N, CI, H, W).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            stmt = "func(data, (kernel_size, kernel_size), (stride, stride), (padding, padding),"\
                              "(dilation, dilation), ceil_mode, return_indices)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data, (kernel_size, kernel_size), (stride, stride), (padding, padding),
                                (dilation, dilation), ceil_mode, return_indices)
            head = torch.ones_like(output)

            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.max_pool2d)
        forward_us, backward_us = test_implementation(ext_quantization.act_quantized_max_pool2d)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_upsample_memory():
    """Test the memory usage"""
    # arguments and test data
    N, H, W, CI = 128, 28, 28, 8
    size, scale_factor, mode, align_corners = None, 2, 'bilinear', False
    data_np = np.random.randn(N, CI, H, W).astype('float32')

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = func(data, size, scale_factor, mode, align_corners)
        output = func(output, size, scale_factor, mode, align_corners)
        output = func(output, size, scale_factor, mode, align_corners)

        return get_memory_usage() - compute_tensor_bytes([output, data])

    usage_ref = test_implementation(F.interpolate)
    print("========== Upsample Memory Test ==========")
    print("Exact.     Usage: %.3f MB" % (usage_ref / 2 ** 20))


def test_bn_correctness():
    # arguments and test data
    N, H, W, CI = 16, 28, 28, 256
    data_np = np.random.randn(N, CI, H, W).astype('float32') * 0.01
    running_mean_np = np.random.randn(CI).astype('float32')
    running_var_np = np.random.randn(CI).astype('float32')
    bn_weight_np = np.random.randn(CI).astype('float32')
    bn_bias_np = np.random.randn(CI).astype('float32')
    training = False

    bn_scheme = QBNScheme()
    config.compress_activation = False

    def test_implementation(func):
        torch.manual_seed(0)
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        running_mean = torch.tensor(running_mean_np).to("cuda")
        running_var = torch.tensor(running_var_np).to("cuda")
        bn_weight = torch.tensor(bn_weight_np).to("cuda").requires_grad_()
        bn_bias = torch.tensor(bn_bias_np).to("cuda").requires_grad_()

        if func == F.batch_norm:
            output = func(data, running_mean, running_var, bn_weight, bn_bias, training, 0.1, 1e-5)
        else:
            output = func(data, running_mean, running_var, bn_weight, bn_bias, training, 0.1, 1e-5, bn_scheme)

        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, bn_weight.grad, bn_bias.grad]]

    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.batch_norm)
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(quantized_batch_norm.apply)

    atol = 1e-3
    rtol = 1e-3
    print("========== BN Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol, rtol=rtol)


def test_conv2d_correctness():
    """Test the correctness of computation results"""
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 28, 28, 256, 256, 3, 1, 1, 1, 1

    print("========== Conv2d Correctness Test ==========")
    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(N, CI, H, W).astype(dtype)
        weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype(dtype)
        bias_np = np.random.randn(CO).astype(dtype)

        def test_implementation(func, scheme):
            torch.manual_seed(0)
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            weight = torch.tensor(weight_np).to("cuda").requires_grad_()
            bias = torch.tensor(bias_np).to("cuda").requires_grad_()

            output = func(data, weight, bias, stride, padding, dilation, groups, scheme)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

        config.activation_compression_bits = [16]
        config.initial_bits = 16
        config.perlayer = False
        config.use_gradient = False
        scheme = QScheme(None)

        config.simulate = True
        output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(quantized_conv2d.apply, scheme)
        config.simulate = False
        output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(quantized_conv2d.apply, scheme)

        atol = 1e-2
        rtol = 1e-2
        assert output_ref.dtype == output_us.dtype
        np.testing.assert_allclose(output_ref, output_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol, rtol=rtol)


def test_conv2d_correctness_per_group_only():
    """Test the correctness of computation results

    NOTE: This test will fail on large shapes or low bits.
    To make this test pass, we should disable stochastic noise.
    """

    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 2, 16, 16, 4, 4, 1, 1, 1, 1, 1

    print("========== Conv2d Correctness Test (per group only) ==========")
    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")

        data_np = np.random.randn(N, CI, H, W).astype(dtype)
        weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype(dtype)
        bias_np = np.random.randn(CO).astype(dtype)

        def test_implementation(func, scheme):
            torch.manual_seed(0)
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            weight = torch.tensor(weight_np).to("cuda").requires_grad_()
            bias = torch.tensor(bias_np).to("cuda").requires_grad_()

            output = func(data, weight, bias, stride, padding, dilation, groups, scheme)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

        config.activation_compression_bits = [8]
        config.perlayer = False
        config.use_gradient = False

        config.simulate = True
        output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(quantized_conv2d.apply, None)
        config.simulate = False
        output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(quantized_conv2d.apply, None)

        atol = 1e-1
        rtol = 1e-1
        assert output_ref.dtype == output_us.dtype
        np.testing.assert_allclose(output_ref, output_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol, rtol=rtol)


def test_conv2d_speed():
    """Test the speed of convolution layer"""


    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 128, 28, 28, 256, 256, 3, 1, 1, 1, 1

    print("========== Conv2d Speed Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(N, CI, H, W).astype(dtype)
        weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype(dtype)
        bias_np = np.random.randn(CO).astype(dtype)

        scheme = QScheme(None)

        def test_implementation(func, scheme):
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            weight = torch.tensor(weight_np).to("cuda").requires_grad_()
            bias = torch.tensor(bias_np).to("cuda").requires_grad_()

            if func == quantized_conv2d.apply:
                output = func(data, weight, bias, stride, padding, dilation, groups, scheme)
                stmt = "func(data, weight, bias, stride, padding, dilation, groups, scheme)"
            else:
                output = func(data, weight, bias, stride, padding, dilation, groups)
                stmt = "func(data, weight, bias, stride, padding, dilation, groups)"

            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        config.activation_compression_bits = [16]
        config.initial_bits = 16
        config.perlayer = False
        config.use_gradient = False
        config.simulate = False
        scheme = QScheme(None)

        forward_ref, backward_ref = test_implementation(F.conv2d, None)
        forward_us, backward_us = test_implementation(quantized_conv2d.apply, scheme)

        print("Exact.      forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized.  forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_conv2d_memory_analytical():
    """Compute the memory of activation analytically"""

    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 256, 28, 28, 256, 256, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.randn(CO).astype('float32')
    running_mean = np.zeros((CO,), dtype='float32')
    running_var = np.ones((CO,), dtype='float32')
    bn_weight = np.random.randn(CO).astype('float32')
    bn_bias = np.random.randn(CO).astype('float32')

    scheme = QScheme(num_locations=kernel_size**2)
    bn_scheme = QBNScheme()

    def test_implementation(conv_func, relu_func, bn_func, n_layers=10):
        data = torch.tensor(data_np).to("cuda")

        # allocate input and weights
        data = torch.tensor(data_np).to("cuda").requires_grad_(False)
        weights = []
        running_means = []
        running_vars = []
        bn_weights = []
        bn_biass = []
        for i in range(n_layers):
            weights.append(torch.tensor(weight_np).to("cuda").requires_grad_())
            running_means.append(torch.tensor(running_mean).to("cuda"))
            running_vars.append(torch.tensor(running_var).to("cuda"))
            bn_weights.append(torch.tensor(bn_weight).to("cuda").requires_grad_())
            bn_biass.append(torch.tensor(bn_bias).to("cuda").requires_grad_())

        before_size = get_memory_usage(False)

        # forward n convolution layers
        output = data
        for i in range(n_layers):
            if conv_func == quantized_conv2d.apply:
                output = conv_func(output, weights[i], None, stride, padding, dilation, groups, scheme)
                output = bn_func(output, running_means[i], running_vars[i], bn_weights[i], bn_biass[i], True, 0.1, 1e-5, bn_scheme)
            else:
                output = conv_func(output, weights[i], None, stride, padding, dilation, groups)
                output = bn_func(output, running_means[i], running_vars[i], bn_weights[i], bn_biass[i], True, 0.1, 1e-5)
            output = relu_func(output)

        output = output.sum()

        after_size = get_memory_usage(False)
        output_size = compute_tensor_bytes(output)

        return after_size / 1024**2, (after_size - before_size - output_size) / 1024**2

    total_size_ref, act_size_ref = test_implementation(F.conv2d, lambda x: F.relu(x, inplace=True), F.batch_norm)
    config.simulate = True
    total_size_sim, act_size_sim = test_implementation(quantized_conv2d.apply,
            ext_quantization.act_quantized_relu, quantized_batch_norm.apply)
    config.simulate = False
    total_size_us, act_size_us = test_implementation(quantized_conv2d.apply,
            ext_quantization.act_quantized_relu, quantized_batch_norm.apply)

    print("========== Conv2d Activation Memory Test (bits = %d) ==========" % (config.activation_compression_bits))
    print("Exact.      Total: %7.2f MB\tAct: %7.2f MB" % (total_size_ref, act_size_ref))
    print("Simulation. Total: %7.2f MB\tAct: %7.2f MB" % (total_size_sim, act_size_sim))
    print("Quantized.  Total: %7.2f MB\tAct: %7.2f MB" % (total_size_us, act_size_us))


def test_conv2d_memory_max_batch_size():
    """Find the maximum batch size by gradually increasing the batch size until hitting Out-of-memory error"""

    for device in ["cuda"]:
        def test_implementation(func, n_layers, batch_sizes):
            def run_batch_size(batch_size):
                N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = batch_size, 28, 28, 256, 256, 3, 1, 1, 1, 1
                data_np = np.random.uniform(size=(N, CI, H, W)).astype('float32')
                weight_np = np.random.uniform(size=(CO, CI // groups, kernel_size, kernel_size)).astype('float32')
                bias_np = np.random.uniform(size=(CO,)).astype('float32')

                # allocate input and weights
                data = torch.tensor(data_np).to("cuda").requires_grad_(False)
                weights = []
                for i in range(n_layers):
                    weight = torch.tensor(weight_np).to("cuda").requires_grad_()
                    weights.append(weight)

                before_size = get_memory_usage(False)
    
                # forward n convolution layers
                output = data
                for i in range(n_layers):
                    output = func(output, weights[i], None, stride, padding, dilation, groups)
                output = output.sum()

                after_size = get_memory_usage(False)
                output_size = compute_tensor_bytes(output)
    
                return after_size / 1024**2, (after_size - before_size - output_size) / 1024**2

            # try gradually increased batch sizes
            try:
                for i, batch_size in enumerate(batch_sizes):
                    total_size_ref, act_size_ref = run_batch_size(batch_size)
                    print("batch_size: %4d\t" % batch_size, end="")
                    print("total_memory: %7.2f MB\tact_memory: %7.2f MB" % (total_size_ref, act_size_ref))
            except RuntimeError:
                pass
            finally:
                print("Maximum batch size: %d" % (batch_sizes[i-1]))
       
        print("========== Conv2d Batch Size Test ==========")
        print("---> Exact")
        test_implementation(F.conv2d, n_layers=50, batch_sizes=[100, 200, 250, 300, 350, 400, 450, 500, 1000])
        print("---> Quantized")
        test_implementation(act_quantized_conv2d.apply, n_layers=50, batch_sizes=[100, 200, 250, 500, 1000, 2200, 2300, 2400, 3000, 4000])


if __name__ == "__main__":
    test_relu_correctness()
    test_relu_memory()
    test_relu_speed()

    #test_adaptive_avg_pool2d_correctness()
    #test_adaptive_avg_pool2d_memory()

    #test_max_pool2d_correctness()
    #test_max_pool2d_memory()
    #test_max_pool2d_speed()

    #test_upsample_memory()

    #test_bn_correctness()

    test_conv2d_correctness()
    #test_conv2d_correctness_per_group_only()

    #test_conv2d_speed()

    #config.activation_compression_bits = 2
    #test_conv2d_memory_analytical()

    #config.activation_compression_bits = 2
    #test_conv2d_memory_max_batch_size()
