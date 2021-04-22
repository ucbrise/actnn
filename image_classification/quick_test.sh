python3 main.py --dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 100 -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0 \
    -c quantize --ca=True --cabits=2 --ibits=8 --calg pl \
    --workspace results/tmp --evaluate --training-only \
    --resume results/cifar100/checkpoint-10.pth.tar --resume2 results/cifar100/checkpoint-10.pth.tar  ~/data/cifar100

###### Check cifar memory #####
#for BS in 512
#do
#  python3 main.py --dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 --batch-size $BS --label-smoothing 0 \
#    -c fanin \
#    --workspace results/tmp --gather-checkpoints  ~/data/cifar10
#done

#for BS in 28000
#do
#  python3 main.py --dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 --batch-size $BS --label-smoothing 0 \
#      -c quantize --ca=True --cabits=2 --ibits=8 --calg pl \
#      --workspace results/tmp --gather-checkpoints  ~/data/cifar10
#done

##### Resnet exact ######
#./dist-train 1 0 127.0.0.1 1 resnet50 \
#    "-c fanin" \
#    tmp ~/imagenet 256

##### Resnet quantized ######
#./dist-train 1 0 127.0.0.1 1 resnet152 \
#    "-c quantize --ca=True --cabits=2 --ibits=8 --calg pl" \
#    tmp ~/imagenet 256

