# Image Classficiation

## Requirement
- Put the ImageNet dataset to `~/imagenet`
- Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Train resnet56 on cifar10
```
mkdir -p results/tmp
python3 main.py --dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 10 \
  -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0 \
  --lr 0.1 --momentum 0.9  --warmup 4 \
  -c quantize --ca=True --cabits=2 --ibits=8 --calg pl \
  --workspace results/tmp --gather-checkpoints  ~/data/cifar10
```

## Train resnet50 on imagenet
```
./dist-train 1 0 127.0.0.1 1 resnet50 \
   "-c quantize --ca=True --cabits=2 --ibits=8 --calg pl"\
   tmp ~/imagenet 256
```


## Check gradient variance 
Download model checkpoints
```
wget https://people.eecs.berkeley.edu/~jianfei/results.tar.gz
tar xzvf results.tar.gz
mkdir results/tmp
```

### Cifar 100
```
python3 main.py --dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 100 -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0 \
    -c quantize --ca=True --cabits=2 --ibits=8 --calg pl \
    --workspace results/tmp --evaluate --training-only \
    --resume results/cifar100/checkpoint-10.pth.tar --resume2 results/cifar100/checkpoint-10.pth.tar  ~/data/cifar100
```

| *quantize config* | *Overall Var* | *Val Top1* |
|--------|----------|---------|
| -c quantize --ca=True --cabits=2 --ibits=8 --calg pl | 0.03805697709321976 |  |
