import torch

import actnn
from actnn.conf import config
import actnn.cpp_extension.minimax as ext_minimax
import actnn.cpp_extension.calc_precision as ext_calc_precision


class QScheme(object):
    num_samples = 1
    num_layers = 0
    batch = None
    update_scale = True
    layers = []
    prev_layer = None

    def __init__(self, layer, group=0, num_locations=1, depthwise_groups=1):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits[group]
        if config.use_gradient:
            assert QScheme.num_samples > 1
            self.scales = torch.zeros(QScheme.num_samples)
        else:
            self.scales = torch.tensor([0.0])
        QScheme.layers.append(self)
        self.C = None
        self.dim = None
        self.num_locations = num_locations      # Kernel size
        self.depthwise_groups = depthwise_groups    # Depthwise separable conv
        self.layer = layer
        self.group = group

        # debug
        self.name = 'layer_{}'.format(QScheme.num_layers)
        QScheme.num_layers += 1

    def get_scale(self):
        if config.use_gradient:
            assert QScheme.batch is not None
            scale = self.scales[QScheme.batch].clone()
            avg_scale = scale.mean()
            scale[scale == 0] = avg_scale + 1e-9
            return scale
        else:
            return self.scales

    def set_scale(self, grad):
        if QScheme.update_scale:
            if config.use_gradient:
                assert QScheme.batch is not None
                scale = grad.view(grad.shape[0], -1).float().norm(dim=1).square().cpu()
                self.scales[QScheme.batch] = self.scales[QScheme.batch] * 0.5 + scale * 0.5
            else:
                scale = grad.view(grad.shape[0], -1).float().norm(dim=1).square()
                self.scales = scale.mean()

    def compute_quantization_bits(self, input):
        QScheme.prev_layer = self
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

        input_groups = input_flatten.view(-1, config.group_size)    # [-1, group_size]
        mn, mx = ext_minimax.minimax(input_groups)
        if not config.pergroup:    # No per group quantization
            mn = torch.ones_like(mn) * mn.min()
            mx = torch.ones_like(mx) * mx.max()

        # Average range over pixels     G * ||R_n||^2 / I
        Range_sqr = torch.norm((mx - mn).view(N, -1), dim=1).float().square() * (config.group_size / num_pixels)

        # greedy
        grad_sum = self.get_scale().cuda()
        C = (self.num_locations / 4 / self.depthwise_groups * Range_sqr * grad_sum)\
            .to(torch.float32).cpu()
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = ext_calc_precision.calc_precision(b, C, w, int(self.bits * N))         # N

        self.C = C
        self.dim = input.numel() // N
        self.b = b

        return input_groups.view(N, -1, config.group_size), b.cuda(), mn.view(N, -1, 1), mx.view(N, -1, 1)

    @staticmethod
    def allocate_perlayer():
        num_groups = len(config.activation_compression_bits)
        for g in range(num_groups):
            layers = [layer for layer in QScheme.layers if layer.group == g]
            L = len(layers)

            if config.activation_compression_bits[g] == config.initial_bits:
                C = torch.tensor([layer.C.sum() for layer in layers])
                w = torch.tensor([layer.dim for layer in layers], dtype=torch.int)
                total_bits = w.sum() * config.activation_compression_bits[g]
                b = torch.ones(L, dtype=torch.int32) * 8
                b = ext_calc_precision.calc_precision(b, C, w, total_bits)

                for i in range(L):
                    layers[i].bits = layers[i].initial_bits = b[i]
            else:
                Cs = [layer.C for layer in layers]
                C = torch.cat(Cs, 0)

                N = Cs[0].shape[0]

                # TODO ???
                Ws = [torch.ones(N, dtype=torch.int32) * layer.dim for layer in layers]
                # Ws = [torch.ones(N, dtype=torch.int32) for layer in layers]
                w = torch.cat(Ws, 0)

                total_bits = w.sum() * config.activation_compression_bits[g]
                b = torch.ones(N * L, dtype=torch.int32) * config.initial_bits
                b = ext_calc_precision.calc_precision(b, C, w, total_bits)
                for i in range(L):
                    bs = b[i*N : (i+1)*N]
                    layers[i].bits = bs.float().mean()

    def if_allocate_perlayer(self):
        if not config.perlayer:
            return
        for layer in QScheme.layers:
            if layer.C is None:
                return

        first_layer = None
        for layer in QScheme.layers:
            if layer.layer.weight.requires_grad:
                first_layer = layer
                break

        # If myself is the last layer, then reallocate bits per layer
        if config.compress_activation and config.training:
            if self == first_layer:
                QScheme.allocate_perlayer()
                actnn.QBNScheme.allocate_perlayer()
