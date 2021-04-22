INSTALL
====

````
cd quantizers
python setup.py install
mkdir results
./train resnet18    # Exact
./train resnet18 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True"   # Our quantized algorithm
````

If the GPU memory is large, multiply `batch-size`, `lr`, and `warmup` by 2. For distributed setting, multiply `batch-size`, `lr` and `warmup` by `number of GPUs / 4`.