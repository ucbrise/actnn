import ast
import os
import warnings

def set_optimization_level(level):
    if level == 'L0':      # Do nothing
        config.compress_activation = False
        config.adaptive_conv_scheme = config.adaptive_bn_scheme = False
    elif level == 'L1':    # 4-bit conv + 32-bit bn
        config.activation_compression_bits = [4]
        config.adaptive_conv_scheme = config.adaptive_bn_scheme = False
        config.enable_quantized_bn = False
    elif level == 'L2':    # 4-bit
        config.activation_compression_bits = [4]
        config.adaptive_conv_scheme = config.adaptive_bn_scheme = False
    elif level == 'L3':   # 2-bit
        pass
    elif level == 'L3.1': # 2-bit + light system optimization
        pass
        config.cudnn_benchmark_conv2d = False
        config.empty_cache_threshold = 0.2
        config.pipeline_threshold = 3 * 1024**3
    elif level == 'L4':    # 2-bit + swap
        pass
        config.swap = True
    elif level == 'L5':    # 2-bit + swap + defragmentation
        config.swap = True
        os.environ['PYTORCH_CACHE_THRESHOLD'] = '256000000'
        warnings.warn("The defragmentation at L5 requires modification of the c++ "
                      "code of PyTorch. You need to compile this special fork of "
                      "PyTorch: https://github.com/merrymercy/pytorch/tree/actnn_exp")
    elif level == 'swap':
        config.swap = True
        config.compress_activation = False
    else:
        raise ValueError("Invalid level: " + level)

class QuantizationConfig:
    def __init__(self):
        self.compress_activation = True
        self.activation_compression_bits = [2, 8, 8]
        self.pergroup = True
        self.perlayer = True
        self.initial_bits = 8
        self.stochastic = True
        self.training = True
        self.group_size = 256
        self.use_gradient = False
        self.adaptive_conv_scheme = True
        self.adaptive_bn_scheme = True
        self.simulate = False
        self.enable_quantized_bn = True

        # Memory management flag
        self.empty_cache_threshold = None
        self.pipeline_threshold = None
        self.cudnn_benchmark_conv2d = True
        self.swap = False

        # Debug related flag
        self.debug_memory_model = ast.literal_eval(os.environ.get('DEBUG_MEM', "False"))
        self.debug_speed = ast.literal_eval(os.environ.get('DEBUG_SPEED', "False"))
        self.debug_memory_op_forward = False
        self.debug_memory_op_backward = False
        self.debug_remove_bn = False
        self.debug_remove_relu = False

config = QuantizationConfig()

