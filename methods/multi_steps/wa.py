import numpy as np
import torch
from torch.nn import functional as F
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy

EPSILON = 1e-8


# init_epoch=200
# init_lr=0.1
# init_milestones=[60,120,170]
# init_lr_decay=0.1
# init_weight_decay=0.0005


# epochs = 170
# lrate = 0.1
# milestones = [60, 100,140]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4
# num_workers=8
# T=2

class WA(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._T = config.T
        self._old_network = None
        self._kd_lamda = None
        if self._incre_type != 'cil':
            raise ValueError('WA is a class incremental method!')

    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def incremental_train(self):
        self._kd_lamda = self._known_classes / self._total_classes
        super().incremental_train()
        if self._cur_task > 0:
            self._network.weight_align(self._total_classes-self._known_classes)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        losses_clf, losses_kd = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            loss_clf = F.cross_entropy(logits[:,:task_end], targets) * (1-self._kd_lamda)
            loss = loss_clf
            losses_clf += loss_clf.item()
            if self._old_network is not None:
                loss_kd = self._KD_loss(logits[:,:task_begin],self._old_network(inputs)[0],self._T) * self._kd_lamda
                loss += loss_kd
                losses_kd += loss_kd.item()

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
    
    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]