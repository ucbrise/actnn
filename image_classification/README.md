# Image Classficiation
Mixed-precision training for ResNet50 v1.5 modified from [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets).

In this example, we use ActNN by manually constructing the model with the memory-saving layers.

Our training logs are available at [Weights & Biases](https://wandb.ai/actnn/projects).

## Requirements
- Put the ImageNet dataset to `~/imagenet`
- Install required packages
```bash
pip install matplotlib tqdm
```
- Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Train Pre-activation ResNet56 on CIFAR10
```
mkdir -p results/tmp
python3 main.py --dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 10 \
  -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0 \
  --lr 0.1 --momentum 0.9  --warmup 4 \
  -c quantize --ca=True --actnn-level L3 \
  --workspace results/tmp --gather-checkpoints  ~/data/cifar10
```

## Train ResNet50 v1.5 on ImageNet (Full Precision)
```
./dist-train 1 0 127.0.0.1 1 resnet50 \
   "-c quantize --ca=True --actnn-level L3"\
   tmp ~/imagenet 256
```

## Train ResNet50 v1.5 on ImageNet (Mixed Precision)
```
./dist-train 1 0 127.0.0.1 1 resnet50 \
   "--amp --dynamic-loss-scale -c quantize --ca=True --actnn-level L3"\
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
    -c quantize --ca=True --actnn-level L3 \
    --workspace results/tmp --evaluate --training-only \
    --resume results/cifar100/checkpoint-10.pth.tar --resume2 results/cifar100/checkpoint-10.pth.tar  ~/data/cifar100
```

| *quantize config* | *Overall Var* | 
|--------|----------|
| -c quantize --ca=True --actnn-level L3 | 0.03805697709321976 | 
