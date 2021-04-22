mkdir results/$2
python main.py --dataset cifar10 --arch $1 --gather-checkpoints --workspace results/$2 --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4 --epochs 200 $3 ~/data/cifar10
