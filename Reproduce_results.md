# Reproduce results

There are several evaluation metrics in continual learning. We compiled some commonly used metrics bellow to reach a agreement on metric names.

We identify $A_{i}^{j}$ ( i,j = 1,2,...,T ) as the accuracy of the model trained after task j and test on the data of task i. $A^{j}$ is the accuracy of the model trained after task j and test on all the seen tasks (test data from task 0, 1, ... , j ). 

$\uparrow$ and $\downarrow$ mean "highter is better" and "lower is better" respectively.

`Avg Acc (AvgACC)` $\uparrow$:
$$Avg ACC = \frac{1}{T}\sum_{i=1}^{T}A^{i}$$

`Final Average Accuracy (FAA)` $\uparrow$:
$$FAA = \frac{1}{T}\sum_{i=1}^{T}A_{i}^{T}$$

`Backward Transfer (BWT)` $\downarrow$:
$$BWT = \frac{1}{T-1}\sum_{i=2}^{T}\frac{1}{i}\sum_{j=1}^{i}(A_{j}^{i}-A_{j}^{j})$$

`Average Forgetting (AvgF)` $\downarrow$:
$$Avg F = \frac{1}{T-1}\sum_{i=1}^{T-1} (\max_{t=1,...,T-1}A_{i}^{t}-A_{i}^{T})$$

The result of the methods may be affected by the incremental order (In my opinion), random seed. You can either generate more orders and average their results or increase the number of training iterations (Or adjust the hyperparameters).

---
## iCaRL: Incremental Classifier and Representation Learning. [[paper](https://arxiv.org/abs/1611.07725)]

Experiment setting (Class incremental): resnet32, cifar100 b0i10, seed 1993, shuffle true, memory_size 2000

Key hyperparameters:
```yaml
T: 2

epochs: 120 #170
batch_size: 128
num_workers: 4

opt_type: sgd
lrate: 0.1
weight_decay: 0.0005
opt_mom: 0.9

scheduler: multi_step
milestones: [49, 63, 90]
lrate_decay: 0.2
```

Reproduce results:
| Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| 88.60  | 79.15  | 74.97  | 68.53  | 65.84  | 61.65  | 58.76  | 55.00  | 52.32  | 49.67   |

Reproduced Average ACC: 65.45

Official Paper Reported Average ACC: 64.1

---

## GEM: Gradient Episodic Memory for Continual Learning. NIPS2017 [[paper](https://arxiv.org/abs/1706.08840)]

Experiment setting (Online task incremental): resnet32, cifar100-b0i10, seed 1993, fixed_memory true, memory_per_class 52

Key hyperparameters:
```yaml
epochs: 1
batch_size: 10
num_workers: 4

opt_type: sgd
lrate: 0.01
```

Reproduce results:
| Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| 40.40  | 41.00  | 48.13  | 48.18  | 55.10  | 58.20  | 60.91  | 63.66  | 64.76  | 65.18   |

Reproduced FAA: 65.18

Official Paper Reported FAA: 65.40

---

## UCIR: Learning a Unified Classifier Incrementally via Rebalancing. CVPR2019[[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)]

Experiment setting (Class incremental): resnet32_cosine, cifar100 b50i10, seed 1993, shuffle true, memory_size 2000

Key hyperparameters:
```yaml
lambda_base: 5 # based on dataset
K: 2 # for all experiments
margin: 0.5 # for all experiments
nb_proxy: 1

epochs: 160
batch_size: 128
num_workers: 4

opt_type: sgd
lrate: 0.1
weight_decay: 0.0005
opt_mom: 0.9

scheduler: multi_step
milestones: [80, 120]
lrate_decay: 0.1
```

Reproduce results:
| Method               | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 |
| -------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| UCIR (CNN) reproduce | 76.86  | 69.62  | 64.64  | 59.26  | 55.92  | 54.41  |
| UCIR (NME) reproduce | 76.44  | 69.23  | 64.24  | 59.50  | 55.67  | 53.54  |

Reproduced Avg ACC: 63.45(CNN), 63.10(NME)

Official Paper Reported Avg ACC: 63.42(CNN), 63.12(NME)

---

## BiC: Large Scale Incremental Learning. [[paper](https://arxiv.org/abs/1905.13260)]

Experiment setting (Class incremental): resnet32, cifar100 b0i10, seed 1993, shuffle true, memory_size 2000

```yaml
key hyperparameters:

```

Reproduce results:
| Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99   |

---

## PODNet: PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning. [[paper](https://arxiv.org/abs/2004.13513)]

Experiment setting (Class incremental): resnet32_cifar, cifar100 b50i10, seed 1993, shuffle true, memory_size 2000

Key hyperparameters:
```yaml
lambda_c_base: 5
lambda_f_base: 1
nb_proxy: 10

layer_names: ['stage_1', 'stage_2', 'stage_3']
            
epochs: 160 # 160
batch_size: 128
num_workers: 4

opt_type: sgd
lrate: 0.1
weight_decay: 0.0005
opt_mom: 0.9

scheduler: cos

epochs_finetune: 20
lrate_finetune: 0.005
```

Reproduce results:
| Method                   | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 |
| ------------------------ | ------ | ------ | ------ | ------ | ------ | ------ |
| PODNet (CNN) reproduce   | 77.78  | 70.70  | 66.36  | 61.61  | 57.57  | 55.30  |
| PODNet (NME) reproduce   | 77.60  | 70.38  | 66.04  | 61.48  | 57.38  | 55.09  |

Reproduced Avg ACC: 64.89(CNN), 64.66(NME)

Official Paper Reported Avg ACC: 64.83(CNN), 64.48(NME)

---

## WA: Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR2020 [[paper](https://arxiv.org/abs/1911.07053)]

Experiment setting (Class incremental): resnet32, cifar100 b0i20, seed 1993, shuffle true, memory_size 2000

Key hyperparameters:
```yaml
T: 2

epochs: 200 #200
batch_size: 128
num_workers: 4

opt_type: sgd
lrate: 0.1
weight_decay: 0.0005
opt_mom: 0.9

scheduler: multi_step
milestones: [60,120,170]
lrate_decay: 0.1
```

Reproduce results:
| Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
| ------ | ------ | ------ | ------ | ------ |
| 81.20  | 73.12  | 66.75  | 60.52  | 55.67  |

Reproduced Avg ACC: 67.45

Official Paper Reported Avg ACC: 66.6

---

## Dark Experience for General Continual Learning: a Strong, Simple Baseline. NeurIPS [[paper](https://arxiv.org/abs/2004.07211)]

Experiment setting (Class incremental): resnet18_cifar, cifar10-b0i2, seed 1993, shuffle false, memory_size 2000

Key hyperparameters:
```yaml
#################
# for dark_er
alpha: 0.3
beta: 0

# for dark_er++
alpha: 0.1
beta: 0.5
#################

epochs: 50 # 170
batch_size: 32
num_workers: 4

opt_type: sgd
lrate: 0.03 # 0.03

scheduler: multi_step
milestones: [35, 45]
lrate_decay: 0.1

```

Reproduce results:
| Method          | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| --------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| DER official    | 89.30  | 77.55  | 69.07  | 66.40  | 62.94  | 61.43  | 56.87  | 55.36  | 53.47  | 51.13   |
| DER reproduce   | 88.00  | 75.90  | 71.27  | 66.75  | 62.78  | 57.65  | 56.21  | 53.55  | 52.29  | 50.81   |
| DER++ official  | 89.90  | 81.05  | 73.80  | 69.50  | 64.90  | 60.95  | 58.89  | 55.63  | 54.7   | 52.11   |
| DER++ reproduce | 88.40  | 79.00  | 74.30  | 68.78  | 63.72  | 60.30  | 59.13  | 56.69  | 54.64  | 53.95   |

---

## DER: Dynamically Expandable Representation for Class Incremental Learning. CVPR2021[[paper](https://arxiv.org/abs/2103.16788)]

Experiment setting (Class incremental): resnet32, cifar100-b0i10, seed 1993

Key hyperparameters:
```yaml
T: 5

epochs: 170 # 170
batch_size: 128
num_workers: 4

opt_type: sgd
lrate: 0.1
weight_decay: 0.0005
opt_mom: 0.9

scheduler: multi_step
milestones: [100, 120, 145]
lrate_decay: 0.1

epochs_finetune: 30 # 200
lrate_finetune: 0.1
milestones_finetune: [15]
```

Reproduce results:
| Method                | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| --------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| DER (w/o P) reproduce | 88.80  | 77.70  | 75.60  | 71.65  | 70.28  | 67.45  | 66.73  | 63.50  | 62.10  | 60.71   |

Reproduced Avg ACC: 70.45

Official Paper Reported Avg ACC: 71.29

---

## Class-Incremental Continual Learning into the eXtended DER-verse. TPAMI 2022 [[paper](https://arxiv.org/abs/2201.00766)]

Experiment setting (Class incremental): resnet18_cifar, cifar100-b0i10, seed 1993, shuffle false, memory_size 2000

```yaml
key hyperparameters:

```

Reproduce results:
| Method          | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| --------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| X-DER official  | 89.10  | 71.10  | 71.40  | 67.95  | 65.92  | 64.13  | 62.40  | 59.64  | 58.17  | 56.96   |
| X-DER reproduce | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99  | 99.99   |

---

## L2P: Learning to Prompt for continual learning. CVPR2022[[paper](https://arxiv.org/abs/2112.08654)]

Experiment setting (Class incremental): ImageNet1K pretrained vit_base_patch16_224, freeze FE, cifar100-b0i10, seed 1993

Key hyperparameters:
```yaml
# shallow_or_deep: True for L2P-shallow, False for L2P-deep
prompt_pool: 30
prompt_length: 20

epochs: 20 #20
batch_size: 64 # 128
num_workers: 4            

opt_type: adam
lrate: 0.001
scheduler: cos
```

Reproduce results:
| Method                | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| --------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| L2P-shallow reproduce | 97.20  | 93.70  | 91.73  | 90.12  | 87.94  | 86.80  | 86.67  | 84.75  | 82.93  | 82.36   |
| L2P-deep reproduce    | 97.50  | 94.65  | 92.97  | 91.22  | 88.44  | 87.58  | 87.57  | 85.12  | 84.37  | 84.10   |

Reproduced FAA: 82.36 (L2P-shallow), 84.10 (L2P-deep)

---

## DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning. ECCV2022 [[paper](https://arxiv.org/abs/2204.04799)]

Experiment setting (Class incremental): ImageNet1K pretrained vit_base_patch16_224, freeze FE, cifar100-b0i10, seed 1993

Key hyperparameters:
```yaml
e_prompt_pool: 10
e_prompt_length: 20
g_prompt_length: 6

epochs: 20 #20
batch_size: 64 # 128
num_workers: 4            

opt_type: adam
lrate: 0.001
scheduler: cos
```

Reproduce results:
| Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| 97.20  | 92.90  | 90.90  | 88.75  | 86.88  | 86.08  | 86.10  | 84.41  | 83.28  | 82.88   |

Reproduced FAA: 82.88

---

## CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning. CVPR2023 [[paper](https://arxiv.org/abs/2211.13218)]

Experiment setting (Class incremental): ImageNet1K pretrained vit_base_patch16_224, freeze FE, cifar100-b0i10, seed 1993

Key hyperparameters:
```yaml
prompt_pool: 100
prompt_length: 8
ortho_weight: 0.005

epochs: 20 #20
batch_size: 64 # 128
num_workers: 4            

opt_type: adam
lrate: 0.001
scheduler: cos
```

Reproduce results:
| Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| 98.40  | 95.70  | 94.07  | 91.52  | 90.26  | 89.40  | 88.93  | 86.68  | 85.12  | 85.13   |

Reproduced FAA: 85.13

Official Paper Reported FAA: 85.16

---

ACL: Adapter Learning in Pretrained Feature Extractor for Continual Learning of Diseases. MICCAI2023 [[paper](https://arxiv.org/abs/2304.09042)]

Experiment setting (Class incremental): ImageNet1K pretrained resnet18, freeze FE, cifar100-b0i10, seed 1993

Key hyperparameters:
```yaml
img_size: 224

layer_names: ['layer1', 'layer2', 'layer3', 'layer4']

epochs: 200 # 200
batch_size: 32
num_workers: 4

opt_type: sgd
lrate: 0.01
weight_decay: 0.0005
opt_mom: 0.9

scheduler: multi_step
milestones: [70, 130, 170]
lrate_decay: 0.1

epochs_finetune: 50 # 50
lrate_finetune: 0.001
milestones_finetune: [15, 35]
```

| Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| 97.00  | 90.80  | 87.80  | 84.42  | 81.94  | 80.70  | 79.31  | 76.28  | 74.60  | 73.23   |

Official Paper Reported Avg ACC: 82.61

Notations: In the paper, we run the experiments with random seed [42, 100, 1993] and the class order is generated by random seed 1993.  