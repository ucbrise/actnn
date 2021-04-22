// Helper for type check
#define CHECK_CUDA_TENSOR_DIM_TYPE(name, n_dim, type)                             \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dim() == n_dim, "The dimension of " #name " is not correct!"); \
  TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!");     \

// Helper for type check
#define CHECK_CUDA_TENSOR_TYPE(name, type)                                        \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!");     \

// Helper for type check
#define CHECK_CUDA_TENSOR_FLOAT(name)                                             \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dtype() == torch::kFloat32 || name.dtype() == torch::kFloat16, \
              "The type of " #name " is not correct!");                           \

// Helper for type check
#define CHECK_CUDA_TENSOR_DIM_FLOAT(name, n_dim)                                  \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dim() == n_dim, "The dimension of " #name " is not correct!"); \
  TORCH_CHECK(name.dtype() == torch::kFloat32 || name.dtype() == torch::kFloat16, \
              "The type of " #name " is not correct!");                           \

