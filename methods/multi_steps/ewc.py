import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.multi_steps.finetune_il import Finetune_IL
from backbone.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

EPSILON = 1e-8


# init_epoch=200
# init_lr=0.1
# init_milestones=[60,120,170]
# init_lr_decay=0.1
# init_weight_decay=0.0005

# epochs = 180
# lrate = 0.1
# milestones = [70, 120,150]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4
# num_workers=4
# T=2
# lamda=1000
# fishermax=0.0001


class EWC(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self.fisher=None    
        
        self._lamda = config.lamda
        self._fishermax = config.fishermax
        if self._incre_type != 'til':
            raise ValueError('EWC is a task incremental method!')

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        losses_clf, losses_ewc = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, feature_outputs = model.forward_til(inputs, self._cur_task)
            
            loss_clf = F.cross_entropy(logits, targets - self._known_classes)
            losses_clf += loss_clf.item()
            if self._cur_task == 0:
                loss = loss_clf
            else:
                loss_ewc = self.compute_ewc()
                losses_ewc += loss_ewc.item()
                loss = loss_clf + self._lamda*loss_ewc

            preds = torch.max(logits, dim=1)[1] + self._known_classes
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', losses_clf/len(train_loader), 'Loss_ewc', losses_ewc/len(train_loader)]
        return model, train_acc, train_loss

    def after_task(self):
        if self.fisher is None:
            self.fisher=self.getFisherDiagonal(self._train_loader)
        else:
            alpha=self._known_classes/self._total_classes
            new_finsher=self.getFisherDiagonal(self._train_loader)
            for n,p in new_finsher.items():
                new_finsher[n][:len(self.fisher[n])]=alpha*self.fisher[n]+(1-alpha)*new_finsher[n][:len(self.fisher[n])]
            self.fisher=new_finsher
        self.mean={n: p.clone().detach() for n, p in self._network.feature_extractor.named_parameters() if p.requires_grad}
        super().after_task()

    def compute_ewc(self):
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.feature_extractor.named_parameters():
                if n in self.fisher.keys():
                    loss += torch.sum((self.fisher[n]) * (p[:len(self.mean[n])] - self.mean[n]).pow(2)) / 2
        else:
            for n, p in self._network.feature_extractor.named_parameters():
                if n in self.fisher.keys():
                    loss += torch.sum((self.fisher[n]) * (p[:len(self.mean[n])] - self.mean[n]).pow(2)) / 2
        return loss
        
    def getFisherDiagonal(self,train_loader):
        fisher = {n: torch.zeros(p.shape).cuda() for n, p in self._network.feature_extractor.named_parameters()
                  if p.requires_grad}
        self._network.train()
        optimizer = optim.SGD(self._network.feature_extractor.parameters(),lr=self._config.lrate)
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = self._network.forward_til(inputs, self._cur_task)
            loss = F.cross_entropy(logits, targets - self._known_classes)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.feature_extractor.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n,p in fisher.items():
            fisher[n]=p/len(train_loader)
            fisher[n]=torch.min(fisher[n],torch.tensor(self._fishermax))
        return fisher