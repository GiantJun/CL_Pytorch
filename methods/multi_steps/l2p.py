import numpy as np
import torch
from methods.multi_steps.finetune_il import Finetune_IL
from torch.nn.functional import cross_entropy

from backbone.inc_net import IncrementalNet
from backbone.l2p_net import L2PNet
from utils.toolkit import accuracy, count_parameters, tensor2numpy

EPSILON = 1e-8

'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

# base is finetune with or without memory_bank
class L2P(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._origin_netork = None

    def prepare_model(self):
        if self._network == None:
            self._network = L2PNet(self._logger, self._config.backbone, self._config.pretrained).cuda()
        if self._incre_type == 'cil':
            self._network.update_fc(self._total_classes)
        elif self._incre_type == 'til':
            self._network.update_til_fc(self._cur_classes)
        if self._config.freeze_fe:
            self._network.freeze_FE()
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
        if self._origin_netork == None:
            self._origin_netork = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained).cuda()

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            if self._incre_type == 'cil':
                logits, feature_outputs = model(inputs)
                loss = cross_entropy(logits, targets)
                preds = torch.max(logits, dim=1)[1]
            elif self._incre_type == 'til':
                logits, feature_outputs = model.forward_til(inputs, self._cur_task)
                loss = cross_entropy(logits, targets - self._known_classes)
                preds = torch.max(logits, dim=1)[1] + self._known_classes
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader)]
        return model, train_acc, train_loss
    
    def _epoch_test(self, model, test_loader, ret_pred_target=False):
        cnn_correct, total = 0, 0
        cnn_pred_all, nme_pred_all, target_all = [], [], []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            if self._incre_type == 'cil':
                outputs, feature_outputs = model(inputs)
                cnn_preds = torch.max(outputs, dim=1)[1]
            elif self._incre_type == 'til':
                outputs, feature_outputs = model.forward_til(inputs, self._cur_task)
                cnn_preds = torch.max(outputs, dim=1)[1] + self._known_classes
                
            if ret_pred_target:
                if self._memory_bank != None and self._apply_nme:
                    nme_pred = self._memory_bank.KNN_classify(feature_outputs['features'])
                    nme_pred_all.append(tensor2numpy(nme_pred))
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
            else:
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            nme_pred_all = np.concatenate(nme_pred_all) if len(nme_pred_all) != 0 else nme_pred_all
            target_all = np.concatenate(target_all)
            return cnn_pred_all, nme_pred_all, target_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc

    def incremental_train(self):
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler, epochs)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        if self._memory_bank != None:
            self._memory_bank.store_samplers(self._sampler_dataset, self._network)