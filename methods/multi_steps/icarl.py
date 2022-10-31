import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

class iCaRL(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        if self._incre_type != 'cil':
            raise ValueError('iCaRL is a class incremental method!')
    
    def prepare_model(self):
        super().prepare_model()
        if self._old_network is not None:
            self._old_network.cuda()
    
    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            onehots = target2onehot(targets, self._total_classes)
            if self._old_network == None:
                loss = binary_cross_entropy_with_logits(logits, onehots)
            else:
                old_onehots = torch.sigmoid(self._old_network(inputs)[0].detach())
                new_onehots = onehots.clone()
                new_onehots[:, :self._known_classes] = old_onehots
                loss = binary_cross_entropy_with_logits(logits, new_onehots)

            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader)]
        return model, train_acc, train_loss