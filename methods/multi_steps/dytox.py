import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim

from backbone.dytox_net import DytoxNet 
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from timm.scheduler.cosine_lr import CosineLRScheduler


EPSILON = 1e-8

class DyTox(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = self._config.T
        self._lamda = self._config.lamda
        self._is_stage2 = False
        self._epoch_sum = 0

        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._milestones_finetune = config.milestones_finetune
        if self._incre_type != 'cil':
            raise ValueError('DyTox is a class incremental method!')

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = DytoxNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
        self._network.update_fc(self._total_classes)
        self._network.freeze_task_tokens(mode='old')
        self._network.freeze_task_heads(mode='old')

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
        if self._old_network is not None:
            self._old_network.cuda()

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self._config.lrate, weight_decay=self._config.weight_decay)
        scheduler = CosineLRScheduler(optimizer, t_initial=self._epochs, lr_min=1e-5, warmup_t=5, warmup_lr_init=1e-6)

        self._is_stage2 = False
        self._epoch_sum = 0
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
            task_id=self._cur_task, epochs=self._epochs, note='stage1')

        if self._cur_task > 0:
            self._logger.info('Finetune the network (classifier part) with the balanced dataset!')
            finetune_train_dataset = self._memory_bank.get_unified_sample_dataset(self._train_dataset, self._network)
            finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self._batch_size,
                                            shuffle=True, num_workers=self._num_workers)
            self._network.freeze_FE()
            self._network.activate_task_tokens(mode='all')
            self._network.activate_task_heads(mode='all')
            ft_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self._lrate_finetune, weight_decay=self._config.weight_decay)
            ft_scheduler = CosineLRScheduler(optimizer, t_initial=self._epochs_finetune, lr_min=1e-5, warmup_t=5, warmup_lr_init=1e-6)
            self._is_stage2 = True
            self._epoch_sum = 0
            self._network = self._train_model(self._network, finetune_train_loader, self._test_loader, ft_optimizer, ft_scheduler,
                task_id=self._cur_task, epochs=self._epochs_finetune, note='stage2')        

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        bce_losses, kd_losses, div_losses = 0., 0., 0.
        correct, total = 0, 0
        model.train()
        if task_id > 0:
            alpha = task_end / task_begin
            clf_factor = 1 - alpha
            kd_factor = alpha
        else:
            clf_factor = 1.
            kd_factor = 0.
                
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, output_features = model(inputs)
            aux_logits = output_features["aux_logits"]

            bce_loss = F.binary_cross_entropy_with_logits(logits, target2onehot(targets, task_end))
            bce_losses += bce_loss.item()
            
            if task_id > 0:
                old_logits, old_output_features = self._old_network(inputs)
                logits_for_distil = logits[:, :task_begin]
                kd_loss =  F.kl_div(
                            F.log_softmax(logits_for_distil / self._T, dim=1),
                            F.log_softmax(old_logits / self._T, dim=1),
                            reduction='mean',
                            log_target=True
                            ) * (self._T ** 2)
                kd_losses += kd_loss.item()
                
                loss = clf_factor*bce_loss + kd_factor*kd_loss

                if not self._is_stage2:
                    # stage 1
                    if self._cur_task > 0:
                        aux_targets = targets.clone()
                        aux_targets = torch.where(aux_targets-task_begin+1>0, aux_targets-task_begin+1, 0)
                        loss_div = F.binary_cross_entropy_with_logits(aux_logits, target2onehot(aux_targets, task_end-task_begin+1))
                        div_losses += loss_div.item()
                        loss += self._lamda*loss_div
            else:
                loss = bce_loss

            preds = torch.max(logits[:,:task_end], dim=1)[1]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)

        if scheduler != None:
            scheduler.step(self._epoch_sum)
        self._epoch_sum += 1
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        if not self._is_stage2: # stage 1
            train_loss = ['Loss', losses/len(train_loader), 'Loss_bce', bce_losses/len(train_loader), 'Loss_kd', kd_losses/len(train_loader), 'Loss_div', div_losses/len(train_loader)]
        else: # stage 2
            train_loss = ['Loss', losses/len(train_loader), 'Loss_bce', bce_losses/len(train_loader), 'Loss_kd', kd_losses/len(train_loader)]
        return model, train_acc, train_loss

# def bce_with_logits(x, y):
#     # print(x.shape[1])
#     # print([y])
#     # print(torch.eye(x.shape[1]))
#     # print(torch.eye(x.shape[1]).to(y.device)[y])
#     # print("++++++++++++++")
#     return F.binary_cross_entropy_with_logits(
#         x,
#         torch.eye(x.shape[1]).to(y.device)[y].to(y.device)
#     )









