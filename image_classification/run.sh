import os

for bbits in [3, 4, 5, 6, 7, 8]:
  for persample in [False, True]:
    for biased in [False, True]:
      CUDA_VISIBLE_DEVICES=0,1 ./test_cifar 2 preact_resnet56 200 "-c quantize --bbits ${bbits} --qa False --persample=True --qw False" 29500
