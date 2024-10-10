# FixMatch
[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685). 的非官方实现
官方实现： [here](https://github.com/google-research/fixmatch).

## Results

### CIFAR10
| #Labels | 40 | 250 | 4000 |
|:---:|:---:|:---:|:---:|
| Paper (RA) | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
| This code | 93.60 | 95.31 | 95.77 |

### CIFAR100
| #Labels | 400 | 2500 | 10000 |
|:---:|:---:|:---:|:---:|
| Paper (RA) | 51.15 ± 1.75 | 71.71 ± 0.11 | 77.40 ± 0.12 |
| This code | 57.50 | 72.93 | 78.12 |

## Usage

### Train
采用 CIFAR-10 dataset 的4000张标注数据训练：

```
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5
```

采用 CIFAR-100 dataset 10000张标注数据 并使用数据并行：
```
python -m torch.distributed.launch --nproc_per_node 4 ./train.py --dataset cifar100 --num-labeled 10000 --arch wideresnet --batch-size 16 --lr 0.03 --wdecay 0.001 --expand-labels --seed 5 --out results/cifar100@10000
```

### Monitoring training progress
```
tensorboard --logdir=<your out_dir>
```

## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)
