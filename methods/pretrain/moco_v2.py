import logging
import os

import torch
from torch.nn import functional as F

from backbone.inc_net import get_backbone
from backbone.mocoV2_net import MoCoNet
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters


class MOCO_v2(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._K = config.K
        self._m = config.m
        self._T = config.T
    
    def prepare_model(self):
        if self._network == None:
            self._network = MoCoNet(get_backbone(self._logger, self._config.backbone), K=self._K, m=self._m, T=self._T, mlp=True)
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        model.train()
        for _, inputs, targets in train_loader:
            inputs[0], inputs[1], targets = inputs[0].cuda(), inputs[1].cuda(), targets.cuda()
            logits, labels = self._network(inputs[0], inputs[1], targets)

            loss = F.cross_entropy(logits, labels) 
    
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

    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = os.path.join(self._logdir, filename)
        if state_dict != None:
            save_dict = state_dict
        else:
            save_dict = model.encoder_q.state_dict()
        torch.save({
            'state_dict': save_dict,
            'config':self._config.get_save_config()
            }, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))


        
        
        
        

        

