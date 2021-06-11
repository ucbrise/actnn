# ActNN : Activation Compressed Training

This is the official project repository for [ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/abs/2104.14129) by
Jianfei Chen\*, Lianmin Zheng\*, Zhewei Yao, Dequan Wang, Ion Stoica, Michael W. Mahoney, and Joseph E. Gonzalez.

**TL; DR.**
ActNN is a PyTorch library for memory-efficient training. It reduces the training memory footprint by compressing the saved activations. ActNN is implemented as a collection of memory-saving layers. These layers have an identical interface to their PyTorch counterparts.

## Abstract
The increasing size of neural network models has been critical for improvements in their accuracy, but device memory is not growing at the same rate. This creates fundamental challenges for training neural networks within limited memory environments. In this work, we propose ActNN, a memory-efficient training framework that stores randomly quantized activations for back propagation. We prove the convergence of ActNN for general network architectures, and we characterize the impact of quantization on the convergence via an exact expression for the gradient variance. Using our theory, we propose novel mixed-precision quantization strategies that exploit the activation's heterogeneity across feature dimensions, samples, and layers. These techniques can be readily applied to existing dynamic graph frameworks, such as PyTorch, simply by substituting the layers. We evaluate ActNN on mainstream computer vision models for classification, detection, and segmentation tasks. On all these tasks, ActNN compresses the activation to 2 bits on average, with negligible accuracy loss. ActNN reduces the memory footprint of the activation by 12×, and it enables training with a 6.6× to 14× larger batch size.

![mem_speed_r50](mem_speed_benchmark/mem_speed_r50.png)
*Batch size vs. training throughput on ResNet-50. Red cross mark means out-of-memory. The shaded yellow region denotes the possible batch sizes with full precision training. ActNN achieves significantly larger maximum batch size over other state-of-the-art systems and displays a nontrivial trade-off curve.*


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

## Examples

### Benchmark Memory Usage and Training Speed
See [mem_speed_benchmark](mem_speed_benchmark/). Please do NOT measure the memory usage by `nvidia-smi`.
`nvidia-smi` reports the size of the memory pool allocated by PyTorch, which can be much larger than the size of acutal used memory.

### Image Classification
See [image_classification](image_classification/)

### Object Detection, Semantic Segmentation, Self-Supervised Learning, ...
Here is the example memory-efficient training for ResNet50, built upon the [OpenMMLab](https://openmmlab.com/) toolkits.
We use ActNN with the default optimization level (L3).
Our training runs are available at [Weights & Biases](https://wandb.ai/actnn).

#### Installation

1. Install [mmcv](https://github.com/DequanWang/actnn-mmcv)
```bash
export MMCV_ROOT=/path/to/clone/actnn-mmcv
git clone https://github.com/DequanWang/actnn-mmcv $MMCV_ROOT
cd $MMCV_ROOT
MMCV_WITH_OPS=1 MMCV_WITH_ORT=0 pip install -e .
```

2. Install [mmdet](https://github.com/DequanWang/actnn-mmdet), [mmseg](https://github.com/DequanWang/actnn-mmseg), [mmssl](https://github.com/DequanWang/actnn-mmssl), ...
```bash
export MMDET_ROOT=/path/to/clone/actnn-mmdet
git clone https://github.com/DequanWang/actnn-mmdet $MMDET_ROOT
cd $MMDET_ROOT
python setup.py develop
```

```bash
export MMSEG_ROOT=/path/to/clone/actnn-mmseg
git clone https://github.com/DequanWang/actnn-mmseg $MMSEG_ROOT
cd $MMSEG_ROOT
python setup.py develop
```

```bash
export MMSSL_ROOT=/path/to/clone/actnn-mmssl
git clone https://github.com/DequanWang/actnn-mmssl $MMSSL_ROOT
cd $MMSSL_ROOT
python setup.py develop
```

#### Single GPU training
```python
cd $MMDET_ROOT
python tools/train.py configs/actnn/faster_rcnn_r50_fpn_1x_coco_1gpu.py
# https://wandb.ai/actnn/detection/runs/ye0aax5s
# ActNN mAP 37.4 vs Official mAP 37.4
python tools/train.py configs/actnn/retinanet_r50_fpn_1x_coco_1gpu.py
# https://wandb.ai/actnn/detection/runs/1x9cwokw
# ActNN mAP 36.3 vs Official mAP 36.5
```

```python
cd $MMSEG_ROOT
python tools/train.py configs/actnn/fcn_r50-d8_512x1024_80k_cityscapes_1gpu.py
# https://wandb.ai/actnn/segmentation/runs/159if8da
# ActNN mIoU 72.9 vs Official mIoU 73.6
python tools/train.py configs/actnn/fpn_r50_512x1024_80k_cityscapes_1gpu.py
# https://wandb.ai/actnn/segmentation/runs/25j9iyv3
# ActNN mIoU 74.7 vs Official mIoU 74.5
```

#### Multiple GPUs training
```python
cd $MMSSL_ROOT
bash tools/dist_train.sh configs/selfsup/actnn/moco_r50_v2_bs512_e200_imagenet_2gpu.py 2
# https://wandb.ai/actnn/mmssl/runs/lokf7ydo
# https://wandb.ai/actnn/mmssl/runs/2efmbuww
# ActNN top1 67.3 vs Official top1 67.7
```

For more detailed guidance, please refer to the docs of [mmcv](https://github.com/DequanWang/actnn-mmcv), [mmdet](https://github.com/DequanWang/actnn-mmdet), [mmseg](https://github.com/DequanWang/actnn-mmseg), [mmssl](https://github.com/DequanWang/actnn-mmssl).

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

## Citation

If the actnn library is helpful in your research, please consider citing our paper:

```bibtex
@article{chen2021actnn,
  title={ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training},
  author={Chen, Jianfei and Zheng, Lianmin and Yao, Zhewei and Wang, Dequan and Stoica, Ion and Mahoney, Michael W and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:2104.14129},
  year={2021}
}
```
