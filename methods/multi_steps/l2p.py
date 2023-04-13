import numpy as np
import torch
from torch import nn
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
        self._gamma = config.gamma # 0.1
        if self._incre_type != 'cil':
            raise ValueError('L2P is a class incremental method!')
        # create class mask
        # class_range = list(range(config.total_class_num))
        # self._class_mask = []
        # for i in range(config.nb_tasks):
        #     start = sum(self._increment_steps[:i])
        #     self._class_mask.append(class_range[start:start+self._increment_steps[i]])

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = L2PNet(self._logger, self._config.backbone,batchwise_prompt=True).cuda()
        self._network.update_fc(self._total_classes)
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        self._network = self._network.cuda()
        for param in self._network.feature_extractor.parameters():
            param.requires_grad = False
        self._network.feature_extractor.eval()

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} requre grad!'.format(name))

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_id=None):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                origin_features = self._network.extract_origin_features(inputs)
            
            logits, feature_outputs = model(inputs, task_id=task_id, cls_features=origin_features)
            
            # here is the trick to mask out classes of non-current tasks
            # mask = self._class_mask[task_id] # (n)
            # not_mask = np.setdiff1d(np.arange(self._config.total_class_num), mask) # (all_class_num - n)
            # not_mask = torch.tensor(not_mask, dtype=torch.int64).to(logits.device) 
            # logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf')) # (b, all_class_num)
            
            loss = cross_entropy(logits, targets)
            loss = loss - self._gamma * feature_outputs['reduce_sim']
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
        train_loss = ['Loss', losses/len(train_loader)]
        return model, train_acc, train_loss
    
    def _epoch_test(self, model, test_loader, ret_pred_target=False, task_id=None):
        cnn_correct, total = 0, 0
        cnn_pred_all, target_all = [], []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                origin_features = self._network.extract_origin_features(inputs)
            
            logits, feature_outputs = model(inputs, task_id=task_id, cls_features=origin_features)
            # here is the trick to mask out classes of non-current tasks
            # mask = self._class_mask[?]
            # not_mask = np.setdiff1d(np.arange(self._config.total_class_num), mask)
            # not_mask = torch.tensor(not_mask, dtype=torch.int64).to(logits.device)
            # logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            
            preds = torch.max(logits, dim=1)[1]    
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(preds))
                target_all.append(tensor2numpy(targets))
            else:
                cnn_correct += preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            return cnn_pred_all, [], target_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc