import numpy as np
import torch
from argparse import ArgumentParser
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')
    return parser

class iCaRL(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = config.T
        if self._incre_type != 'cil':
            raise ValueError('iCaRL is a class incremental method!')
    
    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()
    
    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, kd_losses = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            # # bce loss version implementation
            # onehots = target2onehot(targets, self._total_classes)
            # if self._old_network == None:
            #     loss = binary_cross_entropy_with_logits(logits[:,:task_end], onehots)
            # else:
            #     old_onehots = torch.sigmoid(self._old_network(inputs)[0].detach())
            #     new_onehots = onehots.clone()
            #     new_onehots[:, :task_begin] = old_onehots
            #     loss = binary_cross_entropy_with_logits(logits[:,:task_end], new_onehots)

            # ce loss version implementation
            ce_loss = cross_entropy(logits[:,:task_end], targets)
            if self._old_network is None:
                loss = ce_loss
                ce_losses += ce_loss.item()
            else:
                kd_loss = self._KD_loss(logits[:,:task_begin], self._old_network(inputs)[0], self._T)
                kd_losses += kd_loss.item()
                loss = ce_loss + kd_loss

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
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader), 'Loss_kd', kd_losses/len(train_loader)]
        return model, train_acc, train_loss

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        
        # random shuffle soft (teacher logits)
        # b, dim = soft.shape
        # max_idx = torch.argmax(soft, dim=1)
        # mask = torch.ones_like(soft, dtype=bool)
        # mask[torch.arange(mask.shape[0]) ,max_idx] = False
        # soft[mask] = soft[mask].view(b, dim-1)[:, torch.randperm(dim-1)].view(-1)

        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]