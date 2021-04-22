/*
 * Cuda kernels for quantization and mixed-precision packing
 */

#include <torch/extension.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


using torch::IntArrayRef;
using torch::Tensor;

/****************************************/
/****** Pack/Unpack Mixed Precision *****/
/****************************************/
template <typename scalar_t>
__global__ void compute_scale_mixed_precision_kernel(const int32_t* __restrict__ bits,
                                                     const scalar_t* __restrict__ min,
                                                     const scalar_t* __restrict__ max,
                                                     scalar_t* __restrict__ scale,
                                                     int N,
                                                     int num_groups) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N * num_groups) {
    scale[id] = ((scalar_t)((1 << bits[id / num_groups]) - 1)) / (max[id] - min[id] + 2e-6);
  }
}


template <typename scalar_t>
__global__ void pack_mixed_precision_kernel(const int32_t* __restrict__ bits,
                                            const int32_t* __restrict__ prefix_sum,
                                            const scalar_t* __restrict__ data,
                                            const scalar_t* __restrict__ scale,
                                            const scalar_t* __restrict__ min,
                                            int32_t* __restrict__ packed,
                                            std::pair<uint64_t, uint64_t> seeds,
                                            int N,
                                            int num_groups,
                                            int group_size) {
  extern __shared__ int packed_shared[];

  const int n = blockIdx.y;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int id = (n * num_groups + group_id) * group_size + d;
  const int shared_len = group_size * bits[n] / (sizeof(int32_t) * 8);

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(packed_shared)[threadIdx.x] = make_int2(0, 0);
  }

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, id, seeds.second, &state);
  const float noise = curand_uniform(&state);

  const int val = __float2int_rn(fmax((data[id] - min[n * num_groups + group_id]) * scale[n * num_groups + group_id] + noise - 0.5, 0.0f));
  const int offset = d * bits[n];

  __syncthreads();
  for (int i = 0; i < bits[n]; i++) {
    atomicOr(packed_shared + (offset + i) % shared_len, (1 & (val >> i)) << ((offset + i) / shared_len));
  }
  __syncthreads();

  if (threadIdx.x * 2 < shared_len) {
    const int64_t global_offset = \
          ((int64_t)(n == 0 ? 0 : prefix_sum[n-1]) * num_groups * group_size + bits[n] * group_id * group_size) / (sizeof(int32_t) * 8);
    reinterpret_cast<int2*>(packed)[global_offset/2 + threadIdx.x] = \
                             reinterpret_cast<int2*>(packed_shared)[threadIdx.x];
  }
}

// Pack float16/32 data into int32 bit stream
std::pair<Tensor, Tensor> pack_mixed_precision_cuda(Tensor data,
                                                    Tensor min,
                                                    Tensor max,
                                                    Tensor bits,
                                                    bool stochastic) {
  int N = data.size(0);
  int num_groups = data.size(1);
  int group_size = data.size(2);

  int bits_per_int = sizeof(int32_t) * 8;

  // Compute total bits
  Tensor prefix_sum = torch::cumsum(bits, 0, torch::kInt32);
  int64_t total_bits = ((int64_t) prefix_sum[-1].item<int32_t>()) * num_groups * group_size;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  Tensor packed = torch::empty({(total_bits + bits_per_int - 1) / bits_per_int,}, options);

  // Compute scale
  options = torch::TensorOptions().dtype(data.dtype()).device(data.device());
  Tensor scale = torch::empty({N, num_groups, 1}, options);
  int threads = 256;
  int blocks = (N * num_groups + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "compute_scale_mixed_precision", ([&] {
    compute_scale_mixed_precision_kernel<scalar_t><<<blocks, threads>>>(
      bits.data_ptr<int32_t>(), min.data_ptr<scalar_t>(), max.data_ptr<scalar_t>(),
      scale.data_ptr<scalar_t>(), N, num_groups);
  }));

  // Random number generator
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads);
  }
  TORCH_CHECK(stochastic);

  // Pack
  int max_bit = torch::max(bits).item<int32_t>();
  dim3 block_dim(num_groups, N, 1);
  dim3 thread_dim(group_size, 1, 1);
  TORCH_CHECK(group_size % bits_per_int == 0);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_mixed_precision", ([&] {
    pack_mixed_precision_kernel<<<block_dim, thread_dim, max_bit * group_size * sizeof(int) / bits_per_int>>>(
      bits.data_ptr<int32_t>(), prefix_sum.data_ptr<int32_t>(),
      data.data_ptr<scalar_t>(),
      scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
      packed.data_ptr<int32_t>(),
      rng_engine_inputs,
      N, num_groups, group_size);
  }));

  return std::make_pair(packed, scale);
}

// Unpack int32 bit stream to float16/32 data
template <typename scalar_t>
__global__ void unpack_mixed_precision_kernel(const int32_t* __restrict__ bits,
                                              const int32_t* __restrict__ prefix_sum,
                                              const int32_t* __restrict__ data,
                                              const scalar_t* __restrict__ scale,
                                              const scalar_t* __restrict__ min,
                                              scalar_t* __restrict__ unpacked,
                                              int N,
                                              int num_groups,
                                              int group_size) {
  const int n = blockIdx.y;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int id = (n * num_groups + group_id) * group_size + d;
  const int shared_len = group_size * bits[n] / 32;

  const int64_t global_offset = \
        ((int64_t)(n == 0 ? 0 : prefix_sum[n-1]) * num_groups * group_size + bits[n] * group_id * group_size) / 32;
  const int block_offset = d * bits[n];

  int val = 0;
  for (int i = 0; i < bits[n]; i++) {
    val |= (1 & (data[global_offset + (block_offset + i) % shared_len] >> ((block_offset + i) / shared_len))) << i;
  }

  unpacked[id] = ((scalar_t)val) / scale[n * num_groups + group_id] + min[n * num_groups + group_id];
}

// Unpack int32 bit stream to float16/32 data
Tensor unpack_mixed_precision_cuda(Tensor data,
                                   Tensor bits,
                                   Tensor scale,
                                   Tensor min,
                                   int N,
                                   int num_groups,
                                   int group_size) {
  Tensor prefix_sum = torch::cumsum(bits, 0, torch::kInt32);

  auto options = torch::TensorOptions().dtype(scale.dtype()).device(data.device());
  Tensor unpacked = torch::empty({N, num_groups, group_size}, options);

  dim3 block_dim(num_groups, N, 1);
  dim3 thread_dim(group_size, 1, 1);
  TORCH_CHECK(group_size % 32 == 0);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_mixed_precision", ([&] {
    unpack_mixed_precision_kernel<scalar_t><<<block_dim, thread_dim>>>(
      bits.data_ptr<int32_t>(), prefix_sum.data_ptr<int32_t>(),
      data.data_ptr<int32_t>(),
      scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
      unpacked.data_ptr<scalar_t>(),
      N, num_groups, group_size);
  }));

  return unpacked;
}

/****************************************/
/***** Pack/Unpack Single Precision *****/
/****************************************/
template <typename scalar_t>
__global__ void compute_scale_single_precision_kernel(int32_t bits,
                                                      const scalar_t* __restrict__ min,
                                                      const scalar_t* __restrict__ max,
                                                      scalar_t* __restrict__ scale,
                                                      int N,
                                                      int num_groups) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N * num_groups) {
    scale[id] = ((scalar_t)((1 << bits) - 1)) / (max[id] - min[id] + 2e-6);
  }
}

// Pack float16/32 data into int8 bit stream
template<typename scalar_t, bool boundary_check>
__global__ void pack_single_precision_kernel(int32_t bits,
                                             const scalar_t* __restrict__ data,
                                             const scalar_t* __restrict__ scale,
                                             const scalar_t* __restrict__ min,
                                             int8_t* __restrict__ packed,
                                             std::pair<uint64_t, uint64_t> seeds,
                                             int N,
                                             int num_groups,
                                             int group_size) {
  const int no = blockIdx.y;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int work_per_thread = 8 / bits;
  const int64_t global_thread_id = (int64_t)(no * num_groups + group_id) * group_size + d;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_thread_id, seeds.second, &state);

  uint8_t local_packed = 0;
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }

    const int64_t id = (int64_t)(n * num_groups + group_id) * group_size + d;
    const float noise = curand_uniform(&state);
    const int32_t val = __float2int_rn(fmax((data[id] - min[n * num_groups + group_id]) * scale[n * num_groups + group_id] + noise - 0.5, 0.0f));
    local_packed |= (val << (ni * bits));
  }

  packed[global_thread_id] = local_packed;
}

// Pack float16/32 data into int8 bit stream
std::pair<Tensor, Tensor> pack_single_precision_cuda(Tensor data,
                                                     Tensor min,
                                                     Tensor max,
                                                     int bits,
                                                     bool stochastic) {
  int N = data.size(0);
  int num_groups = data.size(1);
  int group_size = data.size(2);

  // Compute total bits
  int work_per_thread = 8 / bits;
  TORCH_CHECK(8 % bits == 0);

  int N_round = N + (work_per_thread - N % work_per_thread) % work_per_thread;
  int64_t total_bits = ((int64_t)bits) * (N_round * num_groups * group_size);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(data.device());
  Tensor packed = torch::empty({(total_bits + 8) / 8,}, options);

  // Compute scale
  options = torch::TensorOptions().dtype(data.dtype()).device(data.device());
  Tensor scale = torch::empty({N, num_groups, 1}, options);
  int threads = 256;
  int blocks = (N * num_groups + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "compute_scale_single_precision", ([&] {
    compute_scale_single_precision_kernel<<<blocks, threads>>>(
      bits, min.data_ptr<scalar_t>(), max.data_ptr<scalar_t>(),
      scale.data_ptr<scalar_t>(), N, num_groups);
  }));

  // Random number generator
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads * work_per_thread);
  }
  TORCH_CHECK(stochastic);

  // Pack
  dim3 block_dim(num_groups, (N + work_per_thread - 1) / work_per_thread, 1);
  dim3 thread_dim(group_size, 1, 1);

  if (N % work_per_thread == 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_single_precision", ([&] {
      pack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
        bits,
        data.data_ptr<scalar_t>(),
        scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
        packed.data_ptr<int8_t>(),
        rng_engine_inputs,
        N, num_groups, group_size);
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_single_precision", ([&] {
      pack_single_precision_kernel<scalar_t, true><<<block_dim, thread_dim>>>(
        bits,
        data.data_ptr<scalar_t>(),
        scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
        packed.data_ptr<int8_t>(),
        rng_engine_inputs,
        N, num_groups, group_size);
    }));
  }

  return std::make_pair(packed, scale);
}

// Unpack int32 bit stream to float16/32 data
template<typename scalar_t, bool boundary_check>
__global__ void unpack_single_precision_kernel(int32_t bits,
                                               const int8_t* __restrict__ data,
                                               const scalar_t* __restrict__ scale,
                                               const scalar_t* __restrict__ min,
                                               scalar_t* __restrict__ unpacked,
                                               int N,
                                               int num_groups,
                                               int group_size) {
  const int no = blockIdx.y;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t global_thread_id = (int64_t)(no * num_groups + group_id) * group_size + d;

  int work_per_thread = 8 / bits;

  uint8_t local_packed = data[global_thread_id];
  int mask = ((1 << bits) - 1);
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }

    const int val = (local_packed >> (ni * bits)) & mask;
    const int64_t id = (int64_t)(n * num_groups + group_id) * group_size + d;
    unpacked[id] = ((scalar_t)val) / scale[n * num_groups + group_id] + min[n * num_groups + group_id];
  }
}

// Unpack int32 bit stream to float16/32 data
Tensor unpack_single_precision_cuda(Tensor data,
                                    int bits,
                                    Tensor scale,
                                    Tensor min,
                                    int N,
                                    int num_groups,
                                    int group_size) {
  auto options = torch::TensorOptions().dtype(scale.dtype()).device(data.device());
  Tensor unpacked = torch::empty({N, num_groups, group_size}, options);

  int work_per_thread = 8 / bits;
  TORCH_CHECK(8 % bits == 0);

  // Unpack
  dim3 block_dim(num_groups, (N + work_per_thread - 1) / work_per_thread, 1);
  dim3 thread_dim(group_size, 1, 1);

  if (N % work_per_thread == 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
      unpack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
        bits,
        data.data_ptr<int8_t>(),
        scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
        unpacked.data_ptr<scalar_t>(),
        N, num_groups, group_size);
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
      unpack_single_precision_kernel<scalar_t, true><<<block_dim, thread_dim>>>(
        bits,
        data.data_ptr<int8_t>(),
        scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
        unpacked.data_ptr<scalar_t>(),
        N, num_groups, group_size);
    }));
  }

  return unpacked;
}


/****************************************/
/********** Act Quantized ReLU **********/
/****************************************/
#define ACT_QUANTIZED_RELU_NUM_THREADS 512
// Unpack int32 bit stream to float16/32 data
template <typename scalar_t>
__global__ void act_quantized_relu_forward_kernel(const scalar_t* __restrict__ data,
                                                  int32_t* __restrict__ mask,
                                                  scalar_t* __restrict__ output,
                                                  int N,
                                                  int mask_len) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_offset = blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8);
  __shared__ int mask_shared[ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8)];

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask_shared)[threadIdx.x] = make_int2(0, 0);
  }

  if (id < N) {
    bool bit = data[id] > 0;
    if (bit) {
      output[id] = data[id];
    } else {
      output[id] = 0.0;
    }

    __syncthreads();
    atomicOr(mask_shared + threadIdx.x % shared_len, bit << (threadIdx.x / shared_len));
    __syncthreads();
  }

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask)[global_offset / 2 + threadIdx.x] = reinterpret_cast<int2*>(mask_shared)[threadIdx.x];
  }
}

std::pair<Tensor, Tensor> act_quantized_relu_forward_cuda(Tensor data) {
  int n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int mask_len = (n_elements + sizeof(int32_t) * 8 - 1) / (sizeof(int32_t) * 8);
  Tensor mask = torch::empty({mask_len}, options);
  Tensor output = torch::empty_like(data);

  int threads = ACT_QUANTIZED_RELU_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "act_quantized_relu_forward", ([&] {
    act_quantized_relu_forward_kernel<scalar_t><<<blocks, threads>>>(
      data.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
      n_elements, mask_len);
  }));

  return std::make_pair(output, mask);
}

template <typename scalar_t>
__global__ void act_quantized_relu_backward_kernel(const scalar_t* __restrict__ grad_output,
                                                   int32_t* __restrict__ mask,
                                                   scalar_t* __restrict__ grad_input,
                                                   int N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_offset = blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8);

  if (id < N) {
    bool bit =  (mask[global_offset + threadIdx.x % shared_len] >> (threadIdx.x / shared_len)) & 1;
    if (bit) {
      grad_input[id] = grad_output[id];
    } else {
      grad_input[id] = 0.0;
    }
  }
}


Tensor act_quantized_relu_backward_cuda(Tensor grad_output, Tensor mask) {
  int n_elements = 1;
  for (size_t i = 0; i < grad_output.dim(); ++i) {
    n_elements *= grad_output.size(i);
  }

  int threads = ACT_QUANTIZED_RELU_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  Tensor grad_input = torch::empty_like(grad_output);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "act_quantized_relu_backward", ([&] {
      act_quantized_relu_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), grad_input.data_ptr<scalar_t>(),
        n_elements);
  }));

  return grad_input;
}


/****************************************/
/******** Act Quantized MaxPool2d *******/
/****************************************/
#define ACT_QUANTIZED_MAX_POOL2D_NUM_THREADS 256
template <typename scalar_t>
__global__ void act_quantized_max_pool2d_forward_kernel(const scalar_t* __restrict__ input,
                                                        scalar_t* __restrict__ output,
                                                        int8_t* __restrict__ max_indices,
                                                        int n_elements,
                                                        int N, int C, int H, int W, int H_out, int W_out,
                                                        int KH, int KW, int SH, int SW, int PH, int PW,
                                                        int DH, int DW) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n_elements) {
    int nc = id / (H_out * W_out);
    int h = id / W_out % H_out;
    int w = id % W_out;

    int h_base = h * SH - PH;
    int h_start = std::max(h_base, 0);
    int h_end = std::min(h_base + KH, H);
    int w_base = w * SW - PW;
    int w_start = std::max(w_base, 0);
    int w_end = std::min(w_base + KW, W);

    scalar_t v = -1e10;
    int8_t index;
    for (int i = h_start; i < h_end; i++) {
        for (int j = w_start; j < w_end; j++) {
            if (input[nc * (H * W) + i * W + j] > v) {
                v = input[nc * (H * W) + i * W + j];
                index = (i - h_base) * KW + j - w_base;
            }
        }
    }

    output[id] = v;
    max_indices[id] = index;
  }
}

std::pair<Tensor, Tensor> act_quantized_max_pool2d_forward_cuda(Tensor input,
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices) {
  int N = input.size(0);
  int C = input.size(1);
  int H = input.size(2);
  int W = input.size(3);
  int H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
  int W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
  auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  Tensor output = torch::empty({N, C, H_out, W_out}, options);
  options = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
  Tensor max_indices = torch::empty({N, C, H_out, W_out}, options);

  int threads = ACT_QUANTIZED_MAX_POOL2D_NUM_THREADS;
  int n_elements = N * C * H_out * W_out;
  int blocks = (n_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "act_quantized_max_pool2d_forward", ([&] {
    act_quantized_max_pool2d_forward_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), max_indices.data_ptr<int8_t>(), n_elements,
      N, C, H, W, H_out, W_out, kernel_size[0], kernel_size[1], stride[0], stride[1],
      padding[0], padding[1], dilation[0], dilation[1]);
  }));

  return std::make_pair(output, max_indices);
}

template <typename scalar_t>
__global__ void act_quantized_max_pool2d_backward_kernel(const scalar_t* __restrict__ grad_output,
                                                         int8_t* __restrict__ max_indices,
                                                         scalar_t* __restrict__ grad_input,
                                                         int n_elements,
                                                         int N, int C, int H, int W, int H_out, int W_out,
                                                         int KH, int KW, int SH, int SW, int PH, int PW,
                                                         int DH, int DW) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n_elements) {
    int nc = id / (H_out * W_out);
    int h = id / W_out % H_out;
    int w = id % W_out;

    int h_base = h * SH - PH;
    int w_base = w * SW - PW;
    int8_t index = max_indices[id];
    int h_offset = index / KW;
    int w_offset = index % KW;

    atomicAdd(grad_input + (nc * H * W) + (h_base + h_offset) * W + (w_base + w_offset), grad_output[id]);
  }
}

Tensor act_quantized_max_pool2d_backward_cuda(Tensor grad_output, Tensor max_indices,
        IntArrayRef input_shape, 
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices) {
  auto options = torch::TensorOptions().dtype(grad_output.dtype()).device(grad_output.device());
  Tensor grad_input =  torch::zeros(input_shape, options);

  int N = grad_output.size(0);
  int C = grad_output.size(1);
  int H_out = grad_output.size(2);
  int W_out = grad_output.size(3);
  int H = input_shape[2];
  int W = input_shape[3];

  int threads = ACT_QUANTIZED_MAX_POOL2D_NUM_THREADS;
  int n_elements = N * C * H_out * W_out;
  int blocks = (n_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "act_quantized_max_pool2d_backward", ([&] {
    act_quantized_max_pool2d_backward_kernel<scalar_t><<<blocks, threads>>>(
      grad_output.data_ptr<scalar_t>(), max_indices.data_ptr<int8_t>(), grad_input.data_ptr<scalar_t>(),
      n_elements,
      N, C, H, W, H_out, W_out, kernel_size[0], kernel_size[1], stride[0], stride[1],
      padding[0], padding[1], dilation[0], dilation[1]);
  }));

  return grad_input;
}
