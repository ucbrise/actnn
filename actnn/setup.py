from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='actnn',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'actnn.cpp_extension.calc_precision',
              ['actnn/cpp_extension/calc_precision.cc']
          ),
          cpp_extension.CUDAExtension(
              'actnn.cpp_extension.minimax',
              ['actnn/cpp_extension/minimax.cc', 'actnn/cpp_extension/minimax_cuda_kernel.cu']
          ),
          cpp_extension.CUDAExtension(
              'actnn.cpp_extension.backward_func',
              ['actnn/cpp_extension/backward_func.cc']
          ),
          cpp_extension.CUDAExtension(
              'actnn.cpp_extension.quantization',
              ['actnn/cpp_extension/quantization.cc', 'actnn/cpp_extension/quantization_cuda_kernel.cu']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)
