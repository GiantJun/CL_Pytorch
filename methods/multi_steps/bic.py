import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from backbone.inc_net import IncrementalNetWithBias
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy

# epochs = 170
# lrate = 0.1
# milestones = [60,100,140]
# lrate_decay = 0.1
# batch_size = 128
# split_ratio = 0.1
# T = 2
# weight_decay = 2e-4
# num_workers = 8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')
    parser.add_argument('--split_ratio', type=float, default=None, help='split ratio for bic')
    return parser

class BiC(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._T = config.T
        self._split_ratio = config.split_ratio
        self._old_network = None
        if self._incre_type != 'cil':
            raise ValueError('BiC is a class incremental method!')
        
        self._cur_stage = 'stage1'

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        if self._cur_task > 0:
            self._train_dataset, self._valid_dataset, self._sampler_dataset = data_manager.get_dataset_with_split(
                        indices=np.arange(self._known_classes, self._total_classes),
                        source='train', mode='train',
                        appendent=self._memory_bank.get_memory(),
                        val_samples_per_class=int(self._split_ratio * self._memory_size/self._known_classes))

            self._valid_loader = DataLoader(self._valid_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
            self._logger.info('Stage1 dset: {}, Stage2 dset: {}'.format(len(self._train_dataset), len(self._valid_dataset)))
            self.lamda = self._known_classes / self._total_classes
            self._logger.info('Lambda: {:.3f}'.format(self.lamda))
        else:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
            self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')
        
        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')
        
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)      
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)      

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNetWithBias(self._logger, self._config.backbone, self._config.pretrained, bias_correction=True)
        self._network.update_fc(self._total_classes)
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        self._network = self._network.cuda()
        if self._old_network is not None:
            self._old_network = self._old_network.cuda()

    def incremental_train(self):
        self._log_bias_params()
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        
        ### stage1: training ###
        self._cur_stage = 'stage1'
        self._logger.info('-'*10 + ' training stage 1 ' + '-'*10)
        # # Freeze bias layer and train stage1 layer
        # ignored_params = list(map(id, self._network.bias_layers.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
        # network_params = [{'params': base_params, 'lr': self._config.lrate, 'weight_decay': self._config.weight_decay},
        #                   {'params': self._network.bias_layers.parameters(), 'lr': 0, 'weight_decay': 0}]
        
        # optimizer = self._get_optimizer(network_params, self._config, False)

        self._network.freeze_bias_layers()
        # for name, param in self._network.named_parameters():
        #     if param.requires_grad:
        #         self._logger.info('{} requre grad!'.format(name))
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, False)

        scheduler = self._get_scheduler(optimizer, self._config, False)

        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
            task_id=self._cur_task, epochs=epochs, note='stage1')

        if self._cur_task > 0:
            ### stage2: bias correction ###
            self._cur_stage = 'stage2'
            self._logger.info('-'*10 + ' training stage 2 ' + '-'*10)
            self._network.activate_bias_layers()
            # for name, param in self._network.named_parameters():
            #     if param.requires_grad:
            #         self._logger.info('{} requre grad!'.format(name))
            optimizer = self._get_optimizer(self._network.bias_layers[-1].parameters(), self._config, False)
            scheduler = self._get_scheduler(optimizer, self._config, False)

            self._network = self._train_model(self._network, self._valid_loader, self._test_loader, optimizer, scheduler,
                task_id=self._cur_task, epochs=epochs, note='stage2')

        self._log_bias_params()

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        clf_losses, distill_losses = 0., 0.
        correct, total = 0, 0
        # if self._cur_stage == 'stage1':
        #     model.train()
        model.train()
        temp = 0
        for _, inputs, targets in train_loader:
            temp += 1
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)

            if self._cur_stage == 'stage1':
                clf_loss = F.cross_entropy(logits, targets)
                clf_losses += clf_loss.item()
                if self._old_network is not None:
                    old_logits = self._old_network(inputs)[0].detach()
                    hat_pai_k = F.softmax(old_logits / self._T, dim=1)
                    log_pai_k = F.log_softmax(logits[:, :task_begin] / self._T, dim=1)
                    distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))
                    distill_losses += distill_loss.item()
                    loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                else:
                    loss = clf_loss
            elif self._cur_stage == 'stage2':
                loss = F.cross_entropy(torch.softmax(logits, dim=1), targets)
            else:
                raise ValueError('Unknown cur_stage: {}'.format(self._cur_stage))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            preds = torch.max(logits, dim=1)[1]
            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        if self._cur_stage == 'stage1':
            if self._old_network != None:
                train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', clf_losses/len(train_loader), 'Loss_distill', distill_losses/len(train_loader)]
            else:
                train_loss = ['Loss_clf', clf_losses/len(train_loader)]
        else:
            train_loss = ['Loss_clf', losses/len(train_loader)]
        return model, train_acc, train_loss

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def _log_bias_params(self):
        self._logger.info('Parameters of bias layer:')
        params = self._network.get_bias_params()
        for i, param in enumerate(params):
            self._logger.info('{} => {:.3f}, {:.3f}'.format(i, param[0], param[1]))
