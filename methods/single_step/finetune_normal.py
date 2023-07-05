import numpy as np
import torch
from torch.utils.data import DataLoader

from backbone.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.toolkit import accuracy, mean_class_recall, count_parameters, cal_ece, tensor2numpy


EPSILON = 1e-8

'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

# base is finetune with or without memory_bank
class Finetune_normal(BaseLearner):

    def prepare_task_data(self, data_manager):
        # self._cur_task += 1
        # self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._cur_task = data_manager.nb_tasks - 1
        self._cur_classes = data_manager.get_task_size(0)

        self._total_classes = self._known_classes + self._cur_classes
        
        train_dataset = data_manager.get_dataset(source='train', mode='train', indices=np.arange(self._known_classes, self._total_classes))
        test_dataset = data_manager.get_dataset(source='test', mode='test', indices=np.arange(self._known_classes, self._total_classes))
        
        self._logger.info('Train dataset size: {}'.format(len(train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(test_dataset)))

        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        if self._use_valid:
            valid_dataset = data_manager.get_dataset(source='valid', mode='test', indices=np.arange(self._known_classes, self._total_classes))
            self._valid_loader = DataLoader(valid_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            self._logger.info('Valid dataset size: {}'.format(len(valid_dataset)))
        
        self._criterion = self._get_criterion(self._criterion_name)

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
        self._network.update_fc(self._total_classes)
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        if self._config.freeze_fe:
            self._network.freeze_FE()
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, feature_outputs = model(inputs)
            loss = self._criterion(logits, targets)
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

    def _epoch_test(self, model, test_loader, ret_pred_target=False):
        cnn_correct, total = 0, 0
        cnn_pred_all, target_all, features_all = [], [], []
        cnn_max_scores_all = []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feature_outputs = model(inputs)
            cnn_max_scores, cnn_preds = torch.max(torch.softmax(outputs, dim=-1), dim=-1)
            
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                features_all.append(tensor2numpy(feature_outputs['features']))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
            else:
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            features_all = np.concatenate(features_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            return cnn_pred_all, cnn_max_scores_all, target_all, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc

    def eval_task(self):
        self._logger.info(50*"-")
        self._logger.info("log {} of the task".format(self._eval_metric))
        self._logger.info(50*"-")
        cnn_pred, cnn_pred_score, y_true, features = self._epoch_test(self._network, self._test_loader, True)

        if self._eval_metric == 'acc':
            cnn_total, cnn_task = accuracy(cnn_pred.T, y_true, self._total_classes, self._increment_steps)
        elif self._eval_metric == 'mcr':
            cnn_total, cnn_task = mean_class_recall(cnn_pred.T, y_true, self._total_classes, self._increment_steps)
        else:
            raise ValueError('Unknown evaluate metric: {}'.format(self._eval_metric))
        self._logger.info("Final Test Acc: {:.2f}".format(cnn_total))
        self._logger.info("The Expected Calibration Error (ECE) of the model: {:.4f}".format(cal_ece(cnn_pred, cnn_pred_score, y_true)))
        self._logger.info(' ')
        
        self._known_classes = self._total_classes
    