import math

import numpy as np
import torch
from torch.nn import functional as F

from backbone.inc_net import CosineIncrementalNet
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8

# ImageNet1000, ResNet18
'''
epochs = 90
lrate = 0.1
milestones = [30, 60]
lrate_decay = 0.1
batch_size = 128
lamda_base = 10
K = 2
margin = 0.5
weight_decay = 1e-4
num_workers = 16
'''

# CIFAR100, ResNet32
# epochs = 160
# lrate = 0.1
# milestones = [80, 120]
# lrate_decay = 0.1
# batch_size = 128
# lamda_base = 5
# K = 2
# margin = 0.5
# weight_decay = 5e-4
# num_workers = 4


class LUCIR(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._lambda_base = config.lambda_base
        self._K = config.K
        self._margin = config.margin
        self._nb_proxy = config.nb_proxy
        if self._incre_type != 'cil':
            raise ValueError('LUCIR is a class incremental method!')

    def prepare_model(self):
        if self._network == None:
            self._network = CosineIncrementalNet(self._logger, self._config.backbone, self._config.pretrained, 
                        self._config.pretrain_path, nb_proxy=self._nb_proxy)
        self._network.update_fc(self._total_classes, self._cur_task)
        for name, param in self._network.fc.named_parameters():
            if 'fc1' in name:
                param.requires_grad = False
                self._logger.info('{} requires_grad=False'.format(name))
        self._logger.info('Freezing SplitCosineLinear.fc1 ...')
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()

        # Adaptive lambda
        # The definition of adaptive lambda in paper and the official code repository is different.
        # Here we use the definition in official code repository.
        if self._cur_task == 0:
            self.lamda = 0
        else:
            self.lamda = self._lambda_base * math.sqrt(self._known_classes / (self._total_classes - self._known_classes))
        self._logger.info('Adaptive lambda: {}'.format(self.lamda))
        if self._old_network != None:
            self._old_network = self._old_network.cuda()

    def after_task(self):
        super().after_task()
        self._logger.info('Copying current network!')
        self._old_network = self._network.copy().freeze()
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        losses_ce, losses_lf, losses_is = 0., 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            loss_ce = F.cross_entropy(logits, targets)
            losses_ce += loss_ce.item()
            if self._old_network == None:
                loss = loss_ce
            else:
                old_logits, old_feature_outputs = self._old_network(inputs)
                # Less forgetting loss
                loss_lf = F.cosine_embedding_loss(feature_outputs['features'], old_feature_outputs['features'].detach(),
                                torch.ones(inputs.shape[0]).cuda()) * self.lamda
                losses_lf += loss_lf.item()

                loss = loss_ce + loss_lf
                
                # Inter-class speration loss
                old_classes_mask = np.where(tensor2numpy(targets) < self._known_classes)[0]
                if len(old_classes_mask) != 0:
                    scores = feature_outputs['new_scores'][old_classes_mask] # Scores before scaling  (b, nb_new) => (n, nb_new)
                    old_scores = feature_outputs['old_scores'][old_classes_mask] # Scores before scaling  (b, nb_old) => (n, nb_old)

                    # Ground truth targets
                    gt_targets = targets[old_classes_mask] # (n)
                    old_bool_onehot = target2onehot(gt_targets, self._known_classes).type(torch.bool)
                    anchor_positive = torch.masked_select(old_scores, old_bool_onehot)  # (n)
                    anchor_positive = anchor_positive.view(-1, 1).repeat(1, self._K)  # (n, K)

                    # Top K hard
                    anchor_hard_negative = scores.topk(self._K, dim=1)[0]  # (n, K)

                    loss_is = F.margin_ranking_loss(anchor_positive, anchor_hard_negative,
                                                    torch.ones_like(anchor_positive).cuda(), margin=self._margin)
                    losses_is += loss_is.item()

                    loss += loss_is

            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', losses_ce/len(train_loader),
                'Loss_lf', losses_lf/len(train_loader), 'Loss_is', losses_is/len(train_loader)]
        return model, train_acc, train_loss