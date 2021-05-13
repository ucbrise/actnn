import numpy as np
import torch

from actnn.ops import ext_minimax

from timeit_v2 import py_benchmark


def test_minimax_correctness():
    print("========== Minimax Correctness Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(1024, 256).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda")

            if func == torch:
                mn, mx = torch.min(data, 1)[0], torch.max(data, 1)[0]
            else:
                mn, mx = ext_minimax.minimax(data)[:2]

            return [x.detach().cpu().numpy() for x in [mn, mx]]

        mn_ref, mx_ref =  test_implementation(torch)
        mn_us, mx_us = test_implementation(ext_minimax)

        np.testing.assert_allclose(mn_ref, mn_us)
        np.testing.assert_allclose(mx_ref, mx_us)


def test_minimax_speed():
    print("========== Minimax Speed Test ==========")
    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(16384, 256).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda")

            if func == torch:
                stmt = "torch.min(data, 1)[0], torch.max(data, 1)[0]"
            else:
                stmt = "ext_minimax.minimax(data)"

            cost = py_benchmark(stmt, {**globals(), **locals()},
                                setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
            return cost

        cost_ref =  test_implementation(torch)
        cost_us = test_implementation(ext_minimax)

        print("PyTorch.  Cost: %.3f ms" % (cost_ref * 1e3))
        print("Ous.      Cost: %.3f ms" % (cost_us * 1e3))


if __name__ == "__main__":
    test_minimax_correctness()
    test_minimax_speed()

