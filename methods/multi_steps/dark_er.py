import numpy as np
import torch
from torch.nn.functional import mse_loss, cross_entropy
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from backbone.inc_net import IncrementalNet
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--alpha', type=float, default=None, help='balance coeeficient in loss terms')
    parser.add_argument('--beta', type=float, default=None, help='balance coeeficient in loss terms')
    return parser

class Dark_ER(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._alpha = config.alpha
        self._beta = config.beta

        if self._incre_type != 'cil':
            raise ValueError('Dark_ER is a class incremental method!')
    
    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
            self._network.update_fc(self._config.total_class_num)

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            # if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
            #     self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='train')
        
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

    def store_samples(self):
        pass
        # if self._memory_bank != None:
        #     self._memory_bank.store_samples_reservoir(self._train_dataset, self._network)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, kl_losses, buf_ce_losses = 0., 0., 0.
        correct, total = 0, 0
        model.train()
        for idxs, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            ce_loss = cross_entropy(logits, targets)
            loss = ce_loss
            ce_losses += ce_loss.item()

            if not self._memory_bank.is_empty():
            # if task_id > 0:
                buf_inputs, _, buf_soft_targets = self._memory_bank.get_memory_reservoir(self._replay_batch_size,
                                                                                         self._train_dataset.use_path,
                                                                                         self._train_dataset.transform)
                buf_inputs, buf_soft_targets = buf_inputs.cuda(), buf_soft_targets.cuda()

                kl_loss = mse_loss(model(buf_inputs)[0], buf_soft_targets) * self._alpha
                loss += kl_loss
                kl_losses += kl_loss.item()

                if self._beta != 0:
                    buf_inputs, buf_targets, _ = self._memory_bank.get_memory_reservoir(self._replay_batch_size,
                                                                                        self._train_dataset.use_path,
                                                                                        self._train_dataset.transform)
                    buf_inputs, buf_targets = buf_inputs.cuda(), buf_targets.cuda()

                    buf_ce_loss = cross_entropy(model(buf_inputs)[0], buf_targets) * self._beta
                    loss += buf_ce_loss
                    buf_ce_losses += buf_ce_loss.item()
                
            preds = torch.max(logits[:,:task_end], dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)

            self._memory_bank.store_samples_reservoir(self._train_dataset.data[idxs], logits.detach().cpu().numpy(), targets.cpu().numpy())
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader),
            'Loss_kl', kl_losses/len(train_loader)]
        if self._beta != 0:
            train_loss += ['Loss_buf_ce', buf_ce_losses/len(train_loader)]
        return model, train_acc, train_loss