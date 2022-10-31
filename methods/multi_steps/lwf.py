import numpy as np
import torch
from torch.nn.functional import cross_entropy
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy

# init_epoch=200
# init_lr=0.1
# init_milestones=[60,120,160]
# init_lr_decay=0.1
# init_weight_decay=0.0005


# epochs = 250
# lrate = 0.1
# milestones = [60,120, 180,220]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4
# num_workers=8
# T=2
# lamda=3

class LwF(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._init_epoch = config.init_epoch
        self._init_lr = config.init_lr
        self._init_milestones = config.init_milestones
        self._init_lr_decay = config.init_lr_decay
        self._init_weight_decay = config.init_weight_decay

        self._old_network = None
        self._T = config.T
        self._lamda = config.lamda
        if self._incre_type != 'til':
            raise ValueError('LWF is a task incremental method!')

    def _train_model(self, model, train_loader, test_loader):
        if self._old_network is not None:
            self._old_network.cuda()
        return super()._train_model(model, train_loader, test_loader)

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        losses_clf = 0.
        losses_kd = 0.
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            loss_clf = cross_entropy(logits, targets)
            losses_clf += loss_clf.item()
            if self._cur_task == 0:
                loss = loss_clf
            else:
                loss_kd = self._KD_loss(logits[:,:self._known_classes],self._old_network(inputs)[0],self._T)
                losses_kd += loss_kd.item()
                loss = loss_clf + loss_kd

            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', losses_clf/len(train_loader), 'Loss_kd', losses_kd/len(train_loader)]
        return model, train_acc, train_loss

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()
        
    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred/T, dim=1)
        soft = torch.softmax(soft/T, dim=1)
        return -1 * torch.mul(soft, pred).sum()/pred.shape[0]
