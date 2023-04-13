import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch import optim

from backbone.dytox_net import *
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
import utils.utils as cutils
from timm.scheduler.cosine_lr import CosineLRScheduler
from backbone.inc_net import get_backbone


EPSILON = 1e-8

class DyTox(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = self._config.T
        self._lbda = self._config.lbda
        self._individual_classifier = self._config.ind_clf
        self._is_finetuning = False

        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._milestones_finetune = config.milestones_finetune
        if self._incre_type != 'cil':
            raise ValueError('DyTox is a class incremental method!')


    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = get_backbone(self._logger, self._config.backbone, pretrained=self._config.pretrained)
            self._network.head = Classifier(embed_dim=self._network.fc.in_features, nb_base_classes=args.initial_increment)



        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()

        if self._cur_task>0:
            self._network.freeze(['old_task_tokens', 'old_heads'])
            self._logger.info('Freezing old task tokens and old heads !')

        self._network = self._network.cuda()

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def incremental_train(self):
        
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)

        optimazer = optim.AdamW(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self._init_lrate, weight_decay=self._init_weight_decay)
        scheduler = CosineLRScheduler(optimizer, t_initial=self._init_epochs, decay_rate=0.1, lr_min=1e-5, warmup_t=5, warmup_lr_init=1e-6)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs

        self._is_finetuning = False
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
            task_id=self._cur_task, epochs=epochs, note='stage1')

        if self._cur_task > 0:
            self._logger.info('Finetune the network (classifier part) with the balanced dataset!')
            finetune_train_dataset = self._memory_bank.get_unified_sample_dataset(self._train_dataset, self._network)
            finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self._batch_size,
                                            shuffle=True, num_workers=self._num_workers)

            self._network.freeze(['sab'])
            self._network.begin_finetuning()
            ft_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self._init_lrate, weight_decay=self._init_weight_decay)
            ft_scheduler = CosineLRScheduler(optimizer, t_initial=self._epochs_finetune, decay_rate=0.1, lr_min=1e-5, warmup_t=5, warmup_lr_init=1e-6)
            self._is_finetuning = True
            self._network = self._train_model(self._network, finetune_train_loader, self._test_loader, ft_optimizer, ft_scheduler,
                task_id=self._cur_task, epochs=self._epochs_finetune, note='stage2')
            self._network.end_finetuning()

        

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        bce_losses, kd_losses, div_losses = 0., 0., 0.
        correct, total = 0, 0
        model.train()
        teacher_model = self._old_network
        div_output = None
        
    
        
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)

            if isinstance(outputs, dict):
                main_output = outputs['logits']
                div_output = outputs['div']
            else:
                main_output = outputs

            # bce loss
            bce_losses = bce_with_logits(main_output, targets)
            
            if self._cur_task == 0:
                loss = bce_losses
            else:
                # kd loss
                with torch.no_grad():
                    main_output_old = None
                    teacher_outputs = teacher_model(inputs)

                if isinstance(outputs, dict):
                    main_output_old = teacher_outputs['logits']
                else:
                    main_output_old = teacher_outputs
                
                logits_for_distil = main_output[:, :main_output_old.shape[1]]

                alpha = main_output_old.shape[1] / main_output.shape[1]

                clf_factor = 1 - alpha
                kd_factor = alpha

                kd_losses +=  F.kl_div(
                        F.log_softmax(logits_for_distil / self._T, dim=1),
                        F.log_softmax(main_output_old / self._T, dim=1),
                        reduction='mean',
                        log_target=True
                        ) * (self._T ** 2)
                
                # div loss
                # this loss is used for training SAB
                if div_output is not None:
                    div_factor = self._lbda
                    nb_classes = main_output.shape[1]
                    nb_new_classes = div_output.shape[1] - 1
                    nb_old_classes = nb_classes - nb_new_classes

                    div_targets = torch.clone(targets)
                    mask_old_cls = div_targets < nb_old_classes
                    mask_new_cls = ~mask_old_cls

                    div_targets[mask_old_cls] = 0
                    div_targets[mask_new_cls] -= nb_old_classes - 1

                    div_losses = bce_with_logits(div_output, div_targets)

                loss = clf_factor * bce_losses + kd_factor * kd_losses + div_factor * div_losses
            
            preds = torch.max(logits[:,:task_end], dim=1)[1]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)

        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_bce', bce_losses/len(train_loader), 'Loss_kd', kd_losses/len(train_loader), 'Loss_div', div_losses/len(train_loader)]
        return model, train_acc, train_loss










def bce_with_logits(x, y):
    # print(x.shape[1])
    # print([y])
    # print(torch.eye(x.shape[1]))
    # print(torch.eye(x.shape[1]).to(y.device)[y])
    # print("++++++++++++++")
    return F.binary_cross_entropy_with_logits(
        x,
        torch.eye(x.shape[1]).to(y.device)[y].to(y.device)
    )









