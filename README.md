# Pedestrian and Vehicle Detection in Foggy Weather Using Dehazeformer

**Abstract:** 
Recently, the application of target detection in urban transportation has become increasingly widespread.
However, in foggy environments, due to low visibility, targets such as pedestrians and vehicles are not obvious, which easily lead to low detection accuracy and poor robustness. 
We investigate the detection methods in foggy environments, using the DehazeFormer and YOLOv8 as benchmark algorithms.
To address the issue of ambiguous target features in foggy weather,
we introduce an innovative spatial pyramid pooling structure, SimSPPFCSPC, to accelerate network convergence and boost the accuracy and efficiency of the target detection.
To address the issue of detail loss and contextual information loss in foggy targets, we suggest substituting the backbone feature extraction network with EfficientViT network,
implementing a lightweight multi-scale linear attention mechanism to augment the model's resilience and enhance its detection capability.
To address the issue of target size variation in foggy weather, we propose a dynamic IoU-based dynamic adjustment gradient distribution strategy to refine the loss function, bolstering the model's generalization capability.
Experimental results indicate the proposed method not only surpasses the baseline in performance, but also occupies an advantageous position among similar target detection methods, demonstrating excellent detection performance.

## Installation

This implementation is based on [DehazeFormer](https://github.com/IDKiro/DehazeFormer) and [YOLOv8](https://github.com/ultralytics/ultralytics), 
while the former belongs to image dehazing network, and the latter belongs to single-stage target detection network.

```
python 3.8.18
pytorch 1.10.2
torchvision 0.11.3
cuda 11.3
```

1. Create a new conda environment
```
conda create -n dehaze python=3.8
conda activate dehaze
```

2. Install dependencies
```
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

3. Install Ultralytics
```
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
```

## Training and Prediction
During the experiment, we primarily utilize the PASCAL VOC dataset and the Foggy Cityscapes dataset.
We primarily focus on evaluating the dataset after it has been processed by Dehazeformer for dehazing.

### Training
```
yolo task=detect mode=train model=model_name data=dataset_name epochs=300 batch=16
```
Example: model=yolov8-efficientViT-sim-voc-2024.yaml data=VOC.yaml

### Prediction
```
yolo task=detect mode=predict model=weight_path  source=dataset_path  device=cpu
```
Example: model=runs/detect/train/weights/best.pt  source=ultralytics/data/voc
