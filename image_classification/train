mkdir results/$2
python ./multiproc.py --nproc_per_node 4 ./main.py --arch $1 --gather-checkpoints --workspace results/$1 --batch-size 64 --lr 0.256 --warmup 1 $3 ~/data/imagenet
