import logging
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from backbone.inc_net import get_backbone
from backbone.mocoV2_net import MoCoNet
from backbone.simsiam_net import SimSiamNet
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--m', type=float, default=None, help='momenton rate for momenton encoder')
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')
    parser.add_argument('--K', type=int, default=None, help='K for lucir / memory bank size for moco / component for gmm_bayes')
    return parser

class Contrastive_Pretrain(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = config.mode

        self._K = config.K # MoCov2
        self._m = config.m # MoCov2
        self._T = config.T # MoCov2


    
    def prepare_task_data(self, data_manager):
        # self._cur_task += 1
        # self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._cur_task = data_manager.nb_tasks - 1
        self._cur_classes = data_manager.get_task_size(0)

        self._total_classes = self._known_classes + self._cur_classes
        
        train_dataset = data_manager.get_dataset(source='train', mode='train', indices=np.arange(self._known_classes, self._total_classes), two_view=True)
        test_dataset = data_manager.get_dataset(source='test', mode='test', indices=np.arange(self._known_classes, self._total_classes))
        
        self._logger.info('Train dataset size: {}'.format(len(train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(test_dataset)))

        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, drop_last=True)
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, drop_last=True)

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            if 'moco_v2' in self._mode:
                self._network = MoCoNet(self._config.backbone, get_backbone(self._logger, self._config.backbone), K=self._K, m=self._m, T=self._T)
            elif 'simsiam' in self._mode:
                self._network = SimSiamNet(self._config.backbone, get_backbone(self._logger, self._config.backbone))
            else:
                raise ValueError('Unknow contrastive pretrain mode: {}'.format(self._mode))

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        model.train()
        for _, inputs, targets in train_loader:
            inputs[0], inputs[1] = inputs[0].cuda(), inputs[1].cuda()
            loss = self._network(inputs[0], inputs[1])
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        
        if scheduler != None:
            scheduler.step()
        train_loss = ['Loss', losses/len(train_loader)]
        return model, None, train_loss
    
    def eval_task(self):
        return None
    
    def after_task(self):
        self._known_classes = self._total_classes
        if self._save_models:
            self.save_checkpoint('{}_{}_{}_seed{}_epoch{}.pkl'.format(
                    self._config.mode, self._dataset, self._backbone, self._seed, self._config.epochs), 
                    self._network)

    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = os.path.join(self._logdir, filename)
        if 'moco_v2' in self._mode:
            save_dict = model.encoder_q.state_dict()
        elif 'simsiam' in self._mode:
            save_dict = model.encoder.state_dict()
        else:
            raise ValueError('Unknow contrastive pretrain mode: {}'.format(self._mode))
        torch.save({
            'state_dict': save_dict,
            'config':self._config.get_parameters_dict()
            }, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))


        
        
        
        

        

