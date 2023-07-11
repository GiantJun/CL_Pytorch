# CL-Pytorch: Continual Learning Framework for Pytorch

<div align=center>
  <img src="imgs/Learning_and_forgetting.png">
</div>

This codebase implements some SOTA continual / incremental / lifelong learning methods by PyTorch.

By the way, this is also the official repository of [Adapter Learning in Pretrained Feature Extractor for Continual Learning of Diseases. MICCAI2023](https://arxiv.org/abs/2304.09042)

One step baseline method:

- [x] Finetune: Baseline for the upper bound of continual learning which updates parameters with data of all classes available at the same time.

Continual methods already supported:

- [x] Finetune: Baseline method which simply updates parameters when new task data arrive.(with or without memory replay of old class data.)
- [x] iCaRL: Incremental Classifier and Representation Learning. [[paper](https://arxiv.org/abs/1611.07725)]
- [x] GEM: Gradient Episodic Memory for Continual Learning. NIPS2017 [[paper](https://arxiv.org/abs/1706.08840)]
- [x] UCIR: Learning a Unified Classifier Incrementally via Rebalancing. CVPR2019[[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)]
- [x] BiC: Large Scale Incremental Learning. [[paper](https://arxiv.org/abs/1905.13260)]
- [x] PODNet: PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning. [[paper](https://arxiv.org/abs/2004.13513)]
- [x] WA: Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR2020 [[paper](https://arxiv.org/abs/1911.07053)]
- [x] Dark Experience for General Continual Learning: a Strong, Simple Baseline. NeurIPS [[paper](https://arxiv.org/abs/2004.07211)]
- [x] DER: Dynamically Expandable Representation for Class Incremental Learning. CVPR2021[[paper](https://arxiv.org/abs/2103.16788)]
- [x] L2P: Learning to Prompt for continual learning. CVPR2022[[paper](https://arxiv.org/abs/2112.08654)]
- [x] DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning. ECCV2022 [[paper](https://arxiv.org/abs/2204.04799)]
- [x] CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning. CVPR2023 [[paper](https://arxiv.org/abs/2211.13218)]
- [x] ACL: Adapter Learning in Pretrained Feature Extractor for Continual Learning of Diseases. MICCAI2023 [[paper](https://arxiv.org/abs/2304.09042)]

Contrastive model pretraining methods already supported:

- [x] MoCov2: Improved Baselines with Momentum Contrastive Learning. [[paper](https://arxiv.org/abs/2003.04297)]
- [x] SimSiam: Exploring Simple Siamese Representation Learning. [[paper](https://arxiv.org/abs/2011.10566)]

Coming soon:

- [ ] LwF:  Learning without Forgetting. ECCV2016 [[paper](https://arxiv.org/abs/1606.09282)]
- [ ] EWC: Overcoming catastrophic forgetting in neural networks. PNAS2017 [[paper](https://arxiv.org/abs/1612.00796)]
- [ ] LwM: Learning without Memorizing. [[paper](https://arxiv.org/abs/1811.08051)]
- [ ] Layerwise Optimization by Gradient Decomposition for Continual Learning. CVPR2021[[paper](https://arxiv.org/abs/2105.07561v1)]
- [ ] End2End: End-to-End Incremental Learning. [[paper](https://arxiv.org/abs/1807.09536)]
- [ ] FOSTER: Feature Boosting and Compression for Class-incremental Learning. ECCV 2022 [[paper](https://arxiv.org/abs/2204.04662)]
- [ ] Class-Incremental Continual Learning into the eXtended DER-verse. TPAMI 2022 [[paper](https://arxiv.org/abs/2201.00766)]

## How to Use

### Prepare environment

```bash
pip3 install pyyaml tensorboard tensorboard wandb scikit-learn timm quadprog tensorboardX
```

### Run experiments

1. Edit the hyperparameters in the corresponding `options/XXX/XXX.yaml` file

2. Train models:

```bash
python main.py --config options/XXX/XXX.yaml
```

3. Test models with checkpoint (ensure save_model option is True before training)

```bash
python main.py --checkpoint_dir logs/XXX/XXX.pkl
```

If you want to temporary change GPU device in the experiment, you can type `--device #GPU_ID` in terminal without changing 'device' in `.yaml` config file.

### Add datasets and your method

Add corresponding dataset .py file to `datasets/`. It is done! The programme can automatically import the newly added datasets.

we put continual learning methods inplementations in `/methods/multi_steps` folder, pretrain methods in `/methods/pretrain` folder and normal one step training methods in `/methods/singel_steps`.

Supported Datasets:

- Natural image datasets: CIFAR-10, CIFAR-100, ImageNet100, ImageNet1K, ImageNet-R, TinyImageNet, CUB-200

- Medical image datasets: MedMNIST, path16, Skin7, Skin8, Skin40

More information about the supported datasets can be found in `datasets/`

We use `os.environ['DATA']` to access image data. You can config your environment variables in your computer by editing `~/.bashrc` or just change the code.

### Reproduce Results
More details can be found in [Reproduce_results.md](./markdowns/Reproduce_results.md).

## References

We sincerely thank the following works for providing help.

https://github.com/zhchuu/continual-learning-reproduce

https://github.com/G-U-N/PyCIL

https://github.com/GT-RIPL/CODA-Prompt

https://github.com/aimagelab/mammoth

## ToDo

- Results need to be checked: ewc
- Methods need to be modified: mas, lwf
- Multi GPU processing module need to be add.
- A detailed documentation is coming soon
