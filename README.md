# An Classification Framework for Continual Learning

This repository implements some continual / incremental / lifelong learning methods by PyTorch.

One step baseline method:

- [x] Finetune: Baseline for the upper bound of continual learning which updates parameters with data of all classes available at the same time.

Continual methods already supported:

- [x] Finetune: Baseline method which simply updates parameters when new task data arrive.(with or without memory replay of old class data.)
- [x] iCaRL: Incremental Classifier and Representation Learning. [[paper](https://arxiv.org/abs/1611.07725)]
- [x] DER: Dynamically Expandable Representation for Class Incremental Learning. [[paper](https://arxiv.org/abs/2103.16788)]

Coming soon:

- [ ] LwF:  Learning without Forgetting. ECCV2016 [[paper](https://arxiv.org/abs/1606.09282)]
- [ ] EWC: Overcoming catastrophic forgetting in neural networks. PNAS2017 [[paper](https://arxiv.org/abs/1612.00796)]
- [ ] GEM: Gradient Episodic Memory for Continual Learning. NIPS2017 [[paper](https://arxiv.org/abs/1706.08840)]
- [ ] LwM: Learning without Memorizing. [[paper](https://arxiv.org/abs/1811.08051)]
- [ ] End2End: End-to-End Incremental Learning. [[paper](https://arxiv.org/abs/1807.09536)]
- [ ] UCIR: Learning a Unified Classifier Incrementally via Rebalancing. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)]
- [ ] BiC: Large Scale Incremental Learning. [[paper](https://arxiv.org/abs/1905.13260)]
- [ ] PODNet: PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning. [[paper](https://arxiv.org/abs/2004.13513)]
- [ ] WA: Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR2020 [[paper](https://arxiv.org/abs/1911.07053)]
- [ ] FOSTER: Feature Boosting and Compression for Class-incremental Learning. ECCV 2022 [[paper](https://arxiv.org/abs/2204.04662)]

## How to Use

### Run experiment

1. Edit the hyperparameters in the corresponding `options/XXX/XXX.yaml` file

2. Run:

```bash
python main.py --config options/XXX/XXX.yaml
```

If you want to temporary change GPU device in the experiment, you can type `--device #GPU_ID` without changing the `.yaml` config file.

### Add datasets

1. Add corresponding classes to `utils/datasets.py`.
2. Modify the `_get_idata` function in `utils/data_manager.py`.

we put continual learning methods inplementations in `/methods/multi_steps` folder, pretrain methods in `/methods/pretrain` folder and normal one step training methods in `/methods/singel_steps`.

### Results

Exp setting: resnet32, cifar100, seed 1993
| Method Name         | exp seting | Avg Acc | Final Acc | Paper reported Avg Acc |
| ------------------- | ---------- | ------- | --------- | ---------------------- |
| Finetune            | b0i10      | 25.31   | 8.24      | --                     |
| Finetune (Replay)   | b0i10      | 57.68   | 40.24     | --                     |
| iCaRL (NME)         | b0i10      | 64.65   | 48.78     | 64.1                   |
| WA                  | b0i20      | 67.45   | 55.67     | 66.6                   |
| LUCIR               | b50i10     | 64.25   | 54.39     | 63.42                  |
| LUCIR (NME)         | b50i10     | 63.77   | 53.16     | 63.12                  |
| DER (w/o P)         | b0i10      | 72.51   | 62.06     | 71.29                  |
| Joint               | --         | --      | 65.66     | --                     |

`Avg Acc` (Average Incremental Accuracy) is the average of the accuracy after each phase.

## References

We sincerely thank the following works for providing help in our work.

https://github.com/arthurdouillard/incremental_learning.pytorch

https://github.com/zhchuu/continual-learning-reproduce

https://github.com/G-U-N/PyCIL

## ToDo

- Results need to be checked: icarl, podnet, ewc
- Methods need to be modified: bic, gem, mas, lwf
