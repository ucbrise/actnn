import os
import json
import argparse

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


alg_to_config = {
    "exact": "-c fanin",
    "quantize": "-c quantize --ca=True --cabits=2 --ibits=8 --calg pl",
}

network_to_batch_size = {
    "preact_resnet56": [64, 32000],
    "preact_resnet1001": [64, 2048],
    "resnet50": [64, 1024],
    "resnet152": [32, 1024],
}

network_to_command = {
    "preact_resnet56": "python3 main.py --dataset cifar10 --arch preact_resnet56 "
                       "--epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 --batch-size BS "
                       "--training-only --label-smoothing 0 CONFIG "
                       "--workspace results/tmp --gather-checkpoints  ~/data/cifar10",
    "preact_resnet1001": "python3 main.py --dataset cifar10 --arch preact_resnet1001 "
                       "--epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 --batch-size BS "
                       "--training-only --label-smoothing 0 CONFIG "
                       "--workspace results/tmp --gather-checkpoints  ~/data/cifar10",
    "resnet50":  "bash dist-train 1 0 127.0.0.1 1 resnet50 'CONFIG' tmp ~/imagenet BS",
    "resnet152":  "bash dist-train 1 0 127.0.0.1 1 resnet152 'CONFIG' tmp ~/imagenet BS",
}


def run_benchmark(network, alg, batch_size, debug_mem=False, debug_speed=False):
    os.environ['DEBUG_MEM'] = str(debug_mem)
    os.environ['DEBUG_SPEED'] = str(debug_speed)
    cmd = network_to_command[network]
    cmd = cmd.replace("BS", f"{batch_size}").replace("CONFIG", alg_to_config[alg])
    return run_cmd(cmd)


def binary_search_max_batch(network, alg, low, high):
    ret = 0

    while low <= high:
        mid = low + (high - low) // 2
        success = run_benchmark(network, alg, mid, debug_speed=True) == 0
        if success:
            ret = mid
            low = mid + 1
        else:
            high = mid - 1

    return ret


def get_ips(network, alg, batch_size):
    run_benchmark(network, alg, batch_size, debug_speed=True)
    line = list(open("speed_results.tsv").readlines())[-1]
    return json.loads(line)['ips']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
            choices=['linear_scan', 'binary_search'],
            default='linear_scan')
    args = parser.parse_args()

    #networks = ['preact_resnet1001', 'resnet152', 'resnet50',  'preact_resnet56']
    #algs = ['exact', 'quantize']

    networks = ['resnet50']
    algs = ['quantize']
    batch_sizes = [1000]

    if args.mode == 'linear_scan':
        for network in networks: 
            for alg in algs:
                for batch_size in (batch_sizes or network_to_batch_size[network]):
                    if run_benchmark(network, alg, batch_size, debug_mem=False, debug_speed=True) != 0:
                        break
    elif args.mode == 'binary_search':
        for network in networks:
            for alg in algs:
                low, high = network_to_batch_size[network][0], network_to_batch_size[network][-1]
                max_batch_size = binary_search_max_batch(network, alg, low, high)
                ips = get_ips(network, alg, max_batch_size)

                out_file = "max_batch_results.tsv"
                with open(out_file, "a") as fout:
                    val_dict = {
                        "network": network,
                        "algorithm": alg,
                        "max_batch_size": max_batch_size,
                        "ips": ips,
                    }
                    fout.write(json.dumps(val_dict) + "\n")
                print(f"save results to {out_file}")

