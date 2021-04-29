# ActNN : Activation Compressed Training

ActNN is a PyTorch library for memory-efficient training. 
It reduces training memory footprint by compressing the saved activations.
ActNN is implemented as a collection of memory-saving layers.
These layers have identical interface to their PyTorch counterparts.


## Install
- Requirements
```
torch>=1.7.1
torchvision>=0.8.2
```

- Build
```bash
cd actnn
pip install -v -e .
```

## Usage
[mem_speed_benchmark/train.py](mem_speed_benchmark/train.py) is an example on using ActNN for models from torchvision.

### Basic Usage
- Step1: Configure the optimization level  
ActNN provides several optimization levels to control the trade-off between memory saving and computational overhead.
You can set the optimization level by
```python
import actnn
# available choices are ["L0", "L1", "L2", "L3", "L4", "L5"]
actnn.set_optimization_level("L3")
```
See [set_optimization_level](actnn/actnn/conf.py) for more details.

- Step2: Convert the model to use ActNN's layers.  
```python
model = actnn.QModule(model)
```
**Note**:
1. Convert the model _before_ calling `.cuda()`.
2. Set the optimization level _before_ invoking `actnn.QModule` or constructing any ActNN layers.
3. Automatic model conversion only works with standard PyTorch layers.
Please use the modules (`nn.Conv2d`, `nn.ReLU`, etc.), not the functions (`F.conv2d`, `F.relu`).  


- Step3: Print the model to confirm that all the modules (Conv2d, ReLU, BatchNorm) are correctly converted to ActNN layers.
```python
print(model)    # Should be actnn.QConv2d, actnn.QBatchNorm2d, etc.
```
   

### Advanced Features
- Convert the model manually.  
ActNN is implemented as a collection of memory-saving layers, including `actnn.QConv1d, QConv2d, QConv3d, QConvTranspose1d, QConvTranspose2d, QConvTranspose3d, 
    QBatchNorm1d, QBatchNorm2d, QBatchNorm3d, QLinear, QReLU, QSyncBatchNorm, QMaxPool2d`. These layers have identical interface to their PyTorch counterparts. 
You can construct the model manually using these layers as the building blocks.
See `ResNetBuilder` and `resnet_configs` in [image_classification/image_classification/resnet.py](image_classification/image_classification/resnet.py) for example.
- (Optional) Change the data loader  
If you want to use per-sample gradient information for adaptive quantization,
you have to update the dataloader to return sample indices.
You can see `train_loader` in [mem_speed_benchmark/train.py](mem_speed_benchmark/train.py) for example.
In addition, you have to update the configurations.
```python
from actnn import config, QScheme
config.use_gradient = True
QScheme.num_samples = 1300000   # the size of training set
```
You can find sample code in the above script.
- (Beta) Mixed precision training   
ActNN works seamlessly with [Amp](https://github.com/NVIDIA/apex), please see [image_classification](image_classification/) for an example.

## Image Classification
See [image_classification](image_classification/)

## Sementic Segmentation
Will be added later.

## Benchmark Memory Usage and Training Speed
See [mem_speed_benchmark](mem_speed_benchmark/). Please do NOT measure the memory usage with `nvidia-smi`, which could be misleading.

## FAQ
1. Does ActNN supports CPU training?  
Currently, ActNN only supports CUDA.
 
2. Accuracy degradation / diverged training with ActNN.  
ActNN applies lossy compression to the activations. In some challenging cases, our default compression strategy might be too aggressive. 
In this case, you may try more conservative compression strategies (which consume more memory):
    - 4-bit per-group quantization  
   ```python
   actnn.set_optimization_level("L2")
   ```
   - 8-bit per-group quantization
   ```python
   actnn.set_optimization_level("L2")
   actnn.config.activation_compression_bits = [8]
   ```  
    If none of these works, you may report to us by creating an issue.


