# Benchmark Memory Usage and Training Speed on Torchvision Models

## Prepare dataset
Put the ImageNet dataset to `~/imagenet`

## Benchmark Memory Usage
```
DEBUG_MEM=True python3 train.py ~/imagenet --arch ARCH -b BATCH_SIZE --alg ALGORITHM
```

The choices for ARCH are {resnet50, resnet152, wide_resnet101_2, densenet201}  
The choices for ALGORITHM are {exact, actnn-L0, actnn-L1, actnn-L2, actnn-L3, actnn-L4, actnn-L5}  

For example, the command below run actnn-L3 on resnet50
```
DEBUG_MEM=True python3 train.py ~/imagenet --arch resnet50 -b 128 --alg actnn-L3
```

## Benchmark Training Speed
```
DEBUG_SPEED=True python3 train.py ~/imagenet --arch ARCH -b BATCH_SIZE --alg ALGORITHM
```

The choices for ARCH are {resnet50, resnet152, wide_resnet101_2, densenet201}  
The choices for ALGORITHM are {exact, actnn-L0, actnn-L1, actnn-L2, actnn-L3, actnn-L4, actnn-L5}  

For example, the command below run actnn-L3 on resnet50
```
DEBUG_SPEED=True python3 train.py ~/imagenet --arch resnet50 -b 128 --alg actnn-L3
```

