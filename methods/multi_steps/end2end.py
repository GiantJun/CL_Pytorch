import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy

EPSILON = 1e-8

class End2End(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = config.T
        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._milestones_finetune = config.milestones_finetune
        if self._incre_type != 'cil':
            raise ValueError('End2End is a class incremental method!')
    
    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()
    
    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def incremental_train(self):
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        self._is_finetuning = False
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=epochs)

        if self._cur_task > 0:
            self._logger.info('Finetune the network (classifier part) with the balanced dataset!')
            if len(self._multiple_gpus) > 1:
                self._old_network = self._network.module.copy().freeze()
            else:
                self._old_network = self._network.copy().freeze()
            finetune_train_dataset = self._memory_bank.get_unified_sample_dataset(self._train_dataset, self._network)
            finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self._batch_size,
                                            shuffle=True, num_workers=self._num_workers)
            # Update all weights or only the weights of FC layer?
            # According to the experiment results, fine-tuning all weights is slightly better.
            ft_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9, lr=self._lrate_finetune)
            ft_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=ft_optimizer, milestones=self._milestones_finetune, gamma=self._config.lrate_decay)
            self._is_finetuning = True
            self._network = self._train_model(self._network, finetune_train_loader, self._test_loader, ft_optimizer, ft_scheduler, task_id=self._cur_task, epochs=30)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        losses_clf, losses_kd = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            loss_clf = F.cross_entropy(logits, targets)
            losses_clf += loss_clf.item()
            
            if self._cur_task > 0:
                finetuning_task = (self._cur_task + 1) if self._is_finetuning else self._cur_task
                loss_kd = 0.
                old_logits, old_feature_outputs = self._old_network(inputs)
                for i in range(1, finetuning_task+1):
                    lo = sum(self._increment_steps[:i-1])
                    hi = sum(self._increment_steps[:i])

                    task_prob_new = F.softmax(logits[:, lo:hi], dim=1)
                    task_prob_old = F.softmax(old_logits[:, lo:hi], dim=1)

                    task_prob_new = task_prob_new ** (1 / self._T)
                    task_prob_old = task_prob_old ** (1 / self._T)

                    task_prob_new = task_prob_new / task_prob_new.sum(1).view(-1, 1)
                    task_prob_old = task_prob_old / task_prob_old.sum(1).view(-1, 1)

                    loss_kd += F.binary_cross_entropy(task_prob_new, task_prob_old)

                loss_kd *= 1 / finetuning_task
                losses_kd += loss_kd.item()
                loss = loss_clf + loss_kd
            else:
                loss = loss_clf

            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', losses_clf/len(train_loader), 'Loss_kd', losses_kd/len(train_loader)]
        return model, train_acc, train_loss