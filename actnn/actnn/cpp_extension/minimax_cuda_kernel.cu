#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

using torch::Tensor;


__device__ __inline__ c10::Half __shfl_down_sync(const unsigned mask, const c10::Half var,
                                                 const unsigned int delta, const int width) {
  __half var_ = var;
  return __shfl_down_sync(mask, var_, delta, width);
}


__device__ __inline__ c10::Half __shfl_sync(const unsigned mask, const c10::Half var,
                                            const int delta, const int width) {
  __half var_ = var;
  return __shfl_sync(mask, var_, delta, width);
}


template <typename scalar_t>
__global__ void minimax_cuda_kernel(const scalar_t* __restrict__ data,
                                    scalar_t* __restrict__ min,
                                    scalar_t* __restrict__ max,
                                    int64_t N,
                                    int64_t D) {
  scalar_t max_val, min_val;
  max_val = -1e30;
  min_val = 1e30;

  for (int64_t k1_outer = 0; k1_outer < D / 32; ++k1_outer) {
    max_val = std::max(max_val, data[blockIdx.x * D + k1_outer * 32 + threadIdx.x]);
    min_val = std::min(min_val, data[blockIdx.x * D + k1_outer * 32 + threadIdx.x]);
  }

  unsigned int mask;
  scalar_t max_val_t, min_val_t;
  mask = __activemask();

  max_val_t = __shfl_down_sync(mask, max_val, 16, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 8, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 4, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 2, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 1, 32);
  max_val = std::max(max_val, max_val_t);
  max_val = __shfl_sync(mask, max_val, 0, 32);
  max[blockIdx.x] = max_val;

  min_val_t = __shfl_down_sync(mask, min_val, 16, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 8, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 4, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 2, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 1, 32);
  min_val = std::min(min_val, min_val_t);
  min_val = __shfl_sync(mask, min_val, 0, 32);
  min[blockIdx.x] = min_val;
}


std::pair<Tensor, Tensor> minimax_cuda(torch::Tensor data) {
  int64_t N = data.size(0);
  int64_t D = data.size(1);

  auto options = torch::TensorOptions().dtype(data.dtype()).device(data.device());
  Tensor min = torch::empty({N,}, options);
  Tensor max = torch::empty({N,}, options);

  int blocks = N;
  int threads = 32;
  TORCH_CHECK(D % 32 == 0 && D > 32);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "minimax_cuda", ([&] {
    minimax_cuda_kernel<scalar_t><<<blocks, threads>>>(
      data.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(), max.data_ptr<scalar_t>(),
      N, D);
  }));

  return std::make_pair(min, max);
}
