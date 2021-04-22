#include <torch/extension.h>

#include "ext_common.h"

std::pair<torch::Tensor, torch::Tensor> minimax_cuda(torch::Tensor data);

std::pair<torch::Tensor, torch::Tensor> minimax(torch::Tensor data) {
  CHECK_CUDA_TENSOR_FLOAT(data);

  return minimax_cuda(data);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("minimax", &minimax);
}
