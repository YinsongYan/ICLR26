# 
Official Codes for ***Advancing Spatiotemporal Representations in Spiking Neural Networks via Parametric Invertible Transformation (ICLR 2026)***





## 🏊 Usage
## **1. Virtual Environment**
```
# create virtual environment
conda create -n EnhSNN python=3.10
conda activate EnhSNN

# install pytorch=2.0.1, cuda=11.8
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt

# install spikingjelly==0.0.0.0.15
cd sjv15/spikingjelly
pip install .
```

## **2. Data Preparation**


All datasets used in this paper are publicly available. After downloading, please put the dataset in the folder with the corresponding dataset name. 

   1. CIFAR10 and CIFAR100 datasets can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
   2. ImageNet-1k dataset can be downloaded from https://www.image-net.org/download.php
   3. CIFAR10-DVS dataset is available at https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671
   4. DVS-Gesture dataset can be downloaded from https://research.ibm.com/interactive/dvsgesture/





## **3. Model Training**


This section provides instructions on how to train models on various datasets using the provided commands. Follow the steps below for each dataset:


### **(1) CIFAR10**
Run the following commands for training on CIFAR10 dataset (replace the -data_dir with your own path):
```
# for ResNet18 + PIT (D=4)
python train.py -data_dir /datasets/cifar10  -dataset cifar10 -model spiking_resnet18 -T_max 200 -epochs 200  -weight_decay 5e-5  -label_smoothing=0.1 -cutupmix_auto  -neuron EnhLIF

# for ResNet19 + PIT (D=4)
python train_sj.py -data_dir /datasets/cifar10  -dataset cifar10 -model spiking_resnet19 -T_max 200 -epochs 200  -weight_decay 5e-5  -label_smoothing=0.1 -cutupmix_auto  -neuron EnhLIF
``` 


### **(2) CIFAR100**
Run the following commands for training on CIFAR100 dataset (replace the -data_dir with your own path):
```
# for ResNet18 + PIT (D=4)
python train.py -data_dir /datasets/cifar100  -dataset cifar100 -model spiking_resnet18 -T_max 200 -epochs 200  -weight_decay 5e-5  -label_smoothing=0.1 -cutupmix_auto  -neuron EnhLIF

# for ResNet19 + PIT (D=4)
python train_sj.py -data_dir /datasets/cifar100  -dataset cifar100 -model spiking_resnet19 -T_max 200 -epochs 200  -weight_decay 5e-5  -label_smoothing=0.1 -cutupmix_auto  -neuron EnhLIF
``` 

### **(3) ImageNet-1k**
Run the following commands for training on ImageNet-1k dataset (replace the --data-path with your own path), located in the `ImageNet` folder:
``` 
cd ./ImageNet

# for SEW ResNet18 + PIT (D=4)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env train_enh.py --cos_lr_T 200 --model sew_resnet18 --output-dir ./logs --tb --print-freq 256 --amp  --cache-dataset --T 1 --lr 0.01 --wd 1e-4  --epoch 200  --data-path /datasets/imagenet --load ./pretrained/resnet18-f37072fd.pth  -b 256

# for SEW ResNet34 + PIT (D=4)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env train_enh.py --cos_lr_T 200 --model sew_resnet34 --output-dir ./logs --tb --print-freq 256 --amp  --cache-dataset --T 1 --lr 0.01 --wd 1e-4  --epoch 200  --data-path /datasets/imagenet --load ./pretrained/resnet34-b627a593.pth  -b 256
``` 

### **(4) CIFAR10-DVS**
Run the following commands for training on CIFAR10-DVS dataset (replace the -data_dir with your own path):
```
# for VGG11 + PIT (D=4)
python train.py -data_dir /datasets/dvscifar10 -dataset DVSCIFAR10 -T 10 -drop_rate 0.0 -model vggsnn  -T_max 300 -epochs 300 -b 128 -lr=0.05  -tau 0.25 -TET  -neuron EnhLIF
```

### **(5) DVS-Gesture**
Run the following commands for training on DVS-Gesture dataset (replace the -data_dir with your own path):
```
# for VGG11 + PIT (D=4)
python train.py -data_dir /datasets/dvsgesture -dataset dvsgesture -model spiking_vgg11_bn_wdrop  -T 20 -b 16  -T_max 300 -epochs 300  -drop_rate 0.4 -lr=0.05  -TET  -neuron EnhLIF
```







