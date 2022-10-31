import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

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


class BiC(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._T = config.T
        self._split_ratio = config.split_ratio
        self._old_network = None
        if self._incre_type != 'cil':
            raise ValueError('BiC is a class incremental method!')

    def prepare_model(self):
        if self._network == None:
            self._network = IncrementalNetWithBias(self._logger, self._config.backbone, self._config.pretrained, bias_correction=True)
        self._network.update_fc(self._total_classes)
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        if self._cur_task > 0:
            train_dataset, val_dataset = data_manager.get_dataset_with_split(
                        indices=np.arange(self._known_classes, self._total_classes),
                        source='train', mode='train',
                        appendent=self._memory_bank.get_memory(),
                        val_samples_per_class=int(self._split_ratio * self._memory_size/self._known_classes))

            self._val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
            self._logger.info('Stage1 dset: {}, Stage2 dset: {}'.format(len(train_dataset), len(val_dataset)))
            self.lamda = self._known_classes / self._total_classes
            self._logger.info('Lambda: {:.3f}'.format(self.lamda))
        else:
            train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        
        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)


    def incremental_train(self):
        self._log_bias_params()
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        self._stage1_training(self._network, self._train_loader, self._test_loader)
        if self._cur_task > 0:
            self._stage2_bias_correction(self._network, self._val_loader, self._test_loader)

        self._log_bias_params()

    def _epoch_train(self, model, train_loader, optimizer, scheduler, stage):
        losses = 0.
        clf_losses, distill_losses = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)

            if stage == 'training':
                clf_loss = F.cross_entropy(logits, targets)
                clf_losses += clf_loss.item()
                if self._old_network is not None:
                    old_logits = self._old_network(inputs)[0].detach()
                    hat_pai_k = F.softmax(old_logits / self._T, dim=1)
                    log_pai_k = F.log_softmax(logits[:, :self._known_classes] / self._T, dim=1)
                    distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))
                    distill_losses += distill_loss.item()
                    loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                else:
                    loss = clf_loss
            elif stage == 'bias_correction':
                loss = F.cross_entropy(torch.softmax(logits, dim=1), targets)
            else:
                raise NotImplementedError()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            preds = torch.max(logits, dim=1)[1]
            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        if stage == 'training':
            if self._old_network != None:
                train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', clf_losses/len(train_loader), 'Loss_distill', distill_losses/len(train_loader)]
            else:
                train_loss = ['Loss_clf', clf_losses/len(train_loader)]
        else:
            train_loss = ['Loss_clf', losses/len(train_loader)]
        return model, train_acc, train_loss

    def _stage1_training(self, model, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.cuda()
            return
        '''
        self._logger.info('-'*10 + ' training stage 1 ' + '-'*10)
        ignored_params = list(map(id, model.bias_layers.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        network_params = [{'params': base_params, 'lr': self._config.lrate, 'weight_decay': self._config.weight_decay},
                          {'params': model.bias_layers.parameters(), 'lr': 0, 'weight_decay': 0}]
        optimizer = self._get_optimizer(network_params, self._config, False)
        scheduler = self._get_scheduler(optimizer, self._config, False)

        if len(self._multiple_gpus) > 1:
            model = nn.DataParallel(model, self._multiple_gpus)
        model.cuda()
        if self._old_network is not None:
            self._old_network.cuda()
        for epoch in range(self._epochs):
            model, train_acc, train_losses = self._epoch_train(model, train_loader, optimizer, scheduler, stage='training')
            test_acc = self._epoch_test(model, test_loader)
            info = ('Task {}, Epoch {}/{} => '.format(self._cur_task, epoch+1, self._epochs) + 
            ('{} {:.3f}, '* int(len(train_losses)/2)).format(*train_losses) +
            'Train_accy {:.2f}, Test_accy {:.2f}'.format(train_acc, test_acc))
            
            for i in range(int(len(train_losses)/2)):
                self._tblog.add_scalar('seed{}_train/{}'.format(self._seed, train_losses[i*2]), train_losses[i*2+1], self._history_epochs+epoch)
            self._tblog.add_scalar('seed{}_train/Acc'.format(self._seed), train_acc, self._history_epochs+epoch)
            self._tblog.add_scalar('seed{}_test/Acc'.format(self._seed), test_acc, self._history_epochs+epoch)
            self._logger.info(info)
        
        self._history_epochs += self._epochs
        return model

    def _stage2_bias_correction(self, model, val_loader, test_loader):
        self._logger.info('-'*10 + ' training stage 2 ' + '-'*10)
        if isinstance(model, nn.DataParallel):
            model = model.module
        network_params = [{'params': model.bias_layers[-1].parameters(), 'lr': self._config.lrate,
                           'weight_decay': self._config.weight_decay}]
        optimizer = self._get_optimizer(network_params, self._config, False)
        scheduler = self._get_scheduler(optimizer, self._config, False)

        if len(self._multiple_gpus) > 1:
            model = nn.DataParallel(model, self._multiple_gpus)
        model.cuda()

        for epoch in range(self._epochs):
            model, train_acc, train_losses = self._epoch_train(model, val_loader, optimizer, scheduler, stage='bias_correction')
            test_acc = self._epoch_test(model, test_loader)
            info = ('Task {}, Epoch {}/{} => '.format(self._cur_task, epoch+1, self._epochs) + 
            ('{} {:.3f}, '* int(len(train_losses)/2)).format(*train_losses) +
            'Train_accy {:.2f}, Test_accy {:.2f}'.format(train_acc, test_acc))
            
            for i in range(int(len(train_losses)/2)):
                self._tblog.add_scalar('seed{}_train/{}'.format(self._seed, train_losses[i*2]), train_losses[i*2+1], self._history_epochs+epoch)
            self._tblog.add_scalar('seed{}_train/Acc'.format(self._seed), train_acc, self._history_epochs+epoch)
            self._tblog.add_scalar('seed{}_test/Acc'.format(self._seed), test_acc, self._history_epochs+epoch)
            self._logger.info(info)
        
        self._history_epochs += self._epochs
        return model

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        return super().after_task()

    def _log_bias_params(self):
        self._logger.info('Parameters of bias layer:')
        params = self._network.get_bias_params()
        for i, param in enumerate(params):
            self._logger.info('{} => {:.3f}, {:.3f}'.format(i, param[0], param[1]))
