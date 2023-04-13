# An Classification Framework for Continual Learning

This repository implements some continual / incremental / lifelong learning methods by PyTorch.

One step baseline method:

- [x] Finetune: Baseline for the upper bound of continual learning which updates parameters with data of all classes available at the same time.

Continual methods already supported:

- [x] Finetune: Baseline method which simply updates parameters when new task data arrive.(with or without memory replay of old class data.)
- [x] iCaRL: Incremental Classifier and Representation Learning. [[paper](https://arxiv.org/abs/1611.07725)]
- [x] GEM: Gradient Episodic Memory for Continual Learning. NIPS2017 [[paper](https://arxiv.org/abs/1706.08840)]
- [x] UCIR: Learning a Unified Classifier Incrementally via Rebalancing. CVPR2019[[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)]
- [x] PODNet: PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning. [[paper](https://arxiv.org/abs/2004.13513)]
- [x] WA: Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR2020 [[paper](https://arxiv.org/abs/1911.07053)]
- [x] DER: Dynamically Expandable Representation for Class Incremental Learning. CVPR2021[[paper](https://arxiv.org/abs/2103.16788)]
- [x] Layerwise Optimization by Gradient Decomposition for Continual Learning. CVPR2021[[paper](https://arxiv.org/abs/2105.07561v1)]

Contrastive model pretraining methods already supported:

- [x] MoCov2: Improved Baselines with Momentum Contrastive Learning. [[paper](https://arxiv.org/abs/2003.04297)]
- [x] SimSiam: Exploring Simple Siamese Representation Learning. [[paper](https://arxiv.org/abs/2011.10566)]

Coming soon:

- [ ] LwF:  Learning without Forgetting. ECCV2016 [[paper](https://arxiv.org/abs/1606.09282)]
- [ ] EWC: Overcoming catastrophic forgetting in neural networks. PNAS2017 [[paper](https://arxiv.org/abs/1612.00796)]
- [ ] LwM: Learning without Memorizing. [[paper](https://arxiv.org/abs/1811.08051)]
- [ ] End2End: End-to-End Incremental Learning. [[paper](https://arxiv.org/abs/1807.09536)]
- [ ] BiC: Large Scale Incremental Learning. [[paper](https://arxiv.org/abs/1905.13260)]
- [ ] L2P: Learning to Prompt for continual learning. CVPR2022[[paper](https://arxiv.org/abs/2112.08654)]
- [ ] FOSTER: Feature Boosting and Compression for Class-incremental Learning. ECCV 2022 [[paper](https://arxiv.org/abs/2204.04662)]

## How to Use

### Prepare environment

```bash
pip3 install pyyaml tensorboard tensorboard wandb scikit-learn timm quadprog
```

### Run experiments

1. Edit the hyperparameters in the corresponding `options/XXX/XXX.yaml` file

2. Train models:

```bash
python main.py --config options/XXX/XXX.yaml
```

3. Test models with checkpoint

```bash
python main.py --checkpoint_dir logs/XXX/XXX.pkl
```

If you want to temporary change GPU device in the experiment, you can type `--device #GPU_ID` without changing 'device' in `.yaml` config file.

### Add datasets

1. Add corresponding classes to `utils/datasets.py`.
2. Modify the `_get_idata` function in `utils/data_manager.py`.

we put continual learning methods inplementations in `/methods/multi_steps` folder, pretrain methods in `/methods/pretrain` folder and normal one step training methods in `/methods/singel_steps`.

Supported Datasets:

- Natural image datasets: CIFAR-10, CIFAR-100, ImageNet1K, ImageNet100, TinyImageNet

- Medical image datasets: MedMNIST, SD-198

More information about the supported datasets can be found in `utils/dataset.py`

### Results

Exp setting: resnet32, cifar100, seed 1993
| Method Name         | exp seting | Avg Acc | Final Acc | Paper reported Avg Acc |
| ------------------- | ---------- | ------- | --------- | ---------------------- |
| Finetune            | b0i10      | 25.31   | 8.24      | --                     |
| Finetune (Replay)   | b0i10      | 57.68   | 40.24     | --                     |
| Joint               | --         | --      | 65.66     | --                     |
| iCaRL (NME)         | b0i10      | 64.65   | 48.78     | 64.1                   |
| WA                  | b0i20      | 67.45   | 55.67     | 66.6                   |
| LUCIR               | b50i10     | 64.25   | 54.39     | 63.42                  |
| LUCIR (NME)         | b50i10     | 63.77   | 53.16     | 63.12                  |
| DER (w/o P)         | b0i10      | 72.51   | 62.06     | 71.29                  |
| PODNet (CNN)        | b50i10     | 63.91   | 54.24     | 63.19                  |
| PODNet (NME)        | b50i10     | 63.66   | 54.26     | 64.03                  |


`Avg Acc` (Average Incremental Accuracy) is the average of the accuracy after each phase.

## References

We sincerely thank the following works for providing help in our work.

https://github.com/arthurdouillard/incremental_learning.pytorch

https://github.com/zhchuu/continual-learning-reproduce

https://github.com/G-U-N/PyCIL

## ToDo

- Results need to be checked: icarl, podnet, ewc
- Methods need to be modified: bic, gem, mas, lwf
- Multi GPU processing module need to be add.
