from os.path import join

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from backbone.inc_net import IncrementalNet
from methods.multi_steps.finetune_il import Finetune_IL
from utils.replayBank import ReplayBank
from utils.toolkit import accuracy, count_parameters, tensor2numpy, cal_bwf, mean_class_recall, cal_class_avg_acc, cal_ece, cal_openset_test_metrics

EPSILON = 1e-8

'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

class Joint_TIL(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._network_list = []
        if self._incre_type != 'til':
            raise ValueError('Joint_TIL is a task incremental method!')


    def prepare_model(self, checkpoint=None):
        self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
        self._network.update_fc(self._cur_classes)

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")

        self._logger.info("Task {} network's all params: {}".format(self._cur_task, count_parameters(self._network)))
        self._logger.info("Task {} network's trainable params: {}".format(self._cur_task, count_parameters(self._network, True)))
        if len(self._network_list) >= 1:
            self._network_list[-1].freeze()
        self._network = self._network.cuda()
        self._network_list.append(self._network)
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, feature_outputs = model(inputs)
            # if self._incre_type == 'til':
            loss = F.cross_entropy(logits, targets-task_begin)
            preds = torch.max(logits, dim=1)[1] + task_begin
    
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
    
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        cnn_pred_all, nme_pred_all, target_all, features_all = [], [], [], []
        cnn_max_scores_all, nme_max_scores_all = [], []
        self._network_list[task_id].eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = self._network_list[task_id](inputs)

            # if self._incre_type == 'til':
            cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            cnn_preds = cnn_preds + task_begin
                
            if ret_pred_target:
                if self._memory_bank != None and self._apply_nme:
                    nme_pred, nme_max_scores = self._memory_bank.KNN_classify(feature_outputs['features'])
                    nme_pred_all.append(tensor2numpy(nme_pred))
                    nme_max_scores_all.append(tensor2numpy(nme_max_scores))
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                features_all.append(tensor2numpy(feature_outputs['features']))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
            else:
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            nme_pred_all = np.concatenate(nme_pred_all) if len(nme_pred_all) != 0 else nme_pred_all
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            nme_max_scores_all = np.concatenate(nme_max_scores_all) if len(nme_max_scores_all) !=0 else nme_max_scores_all
            target_all = np.concatenate(target_all)
            features_all = np.concatenate(features_all)
            return cnn_pred_all, nme_pred_all, cnn_max_scores_all, nme_max_scores_all, target_all, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                return test_acc, test_acc
            else:
                return test_acc
    
    