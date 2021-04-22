#!/usr/bin/env python
import sys, os

num_nodes = sys.argv[1]
batch_size = 128 // int(num_nodes)
model = sys.argv[2]
dir = sys.argv[3]
epoch = int(sys.argv[4])
options = sys.argv[5]
port = sys.argv[6]

cmd = 'python ./multiproc.py --master_port {port} --nproc_per_node {num_nodes} ./main.py --dataset cifar10 --arch {model} --resume results/{dir}/checkpoint-{epochs}.pth.tar --epochs {epochs_plus_one} --evaluate --raport-file test_raport.json --workspace results/{dir} --batch-size {batch_size} --label-smoothing 0 --weight-decay 1e-4 {options} ~/data/cifar10'.format(
        num_nodes=num_nodes, batch_size=batch_size, model=model, epochs=epoch, epochs_plus_one=epoch+1, options=options, port=port, dir=dir)

print(cmd)
os.system(cmd)

#cmd = 'python compute_std.py results/{} {}'.format(model, num_nodes)
#print(cmd)
#os.system(cmd)

# os.system("python compute_error_std.py")
