# ActNN : Activation Compressed Training

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
- Step1: Convert the model to use ActNN's layers.  
```python
import actnn
model = actnn.QModule(model)
```

- Step2: Configure the optimization level  
ActNN provides several optimization levels to control the trade-off between memory saving and computational overhead.
You can set the optimization level by
```python
# available choices are ["L0", "L1", "L2", "L3", "L4", "L5"]
actnn.set_optimization_level("L3")
```
See [set_optimization_level](actnn/actnn/conf.py) for more details.

### Advanced Features
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


## Image Classification
See [image_classification](image_classification/)

## Sementic Segmentation
Will be added later.

## Benchmark Memory Usage and Training Speed
See [mem_speed_benchmark](mem_speed_benchmark/)

