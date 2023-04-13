import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, mse_loss
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from backbone.inc_net import IncrementalNet
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import cal_bwf, tensor2numpy

EPSILON = 1e-8

class Task_ID_Predict(Finetune_IL):
    
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._task_project_fs = []
        self._task_anchors = []
        self._mode = self._config.mode.split('|')

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path).cuda()
        self._network.freeze_FE()
        task_project_f = nn.Sequential(nn.Linear(self._network.feature_dim, self._network.feature_dim), 
                    nn.ReLU(), nn.Linear(self._network.feature_dim, self._network.feature_dim * 10))
        # task_project_f = nn.Linear(self._network.feature_dim, self._network.feature_dim * 10)
        self._task_project_fs.append(task_project_f.cuda())
    
    def incremental_train(self):
        self._logger.info('-'*10 + ' Computing task {}({}-{}) \'s anchors'.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        task_anchor = []
        for class_id in range(self._known_classes, self._total_classes):
            task_dataset = Subset(self._train_dataset, np.argwhere(self._train_dataset.targets == class_id).squeeze())
            task_loader = DataLoader(task_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
            vectors = []
            with torch.no_grad():
                for _, _inputs, _targets in task_loader:
                    _vectors = self._network.extract_features(_inputs.cuda())
                    vectors.append(_vectors)
            vectors = torch.cat(vectors)
            task_anchor.append(torch.mean(vectors, dim=0))
        self._task_anchors.append(torch.cat(task_anchor))

        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._task_project_fs[-1].parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        self._network = self._train_model(None, self._train_loader, self._test_loader, optimizer, scheduler, self._cur_task, task_id=self._cur_task, epochs=self._epochs)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        self._task_project_fs[-1].train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            features = self._network.feature_extractor(inputs)
            feature_anchors = self._task_project_fs[-1](features)

            # loss = feature_anchors.mm(self._task_anchors[-1].unsqueeze(-1)).mean()
            loss_targets = self._task_anchors[-1].unsqueeze(0).repeat(feature_anchors.shape[0], 1)
            if 'bce_loss' in self._mode:
                loss = binary_cross_entropy(torch.sigmoid(feature_anchors), torch.sigmoid(loss_targets))
            elif 'mse_loss' in self._mode:
                loss = mse_loss(feature_anchors, loss_targets)
            else:
                raise ValueError('Unknown loss in mode: {}'.format(self._mode))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        
        scheduler.step()
        train_loss = ['Loss', losses/len(train_loader)]
        return self._network, 0.0, train_loss
    
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        cnn_pred_all, nme_pred_all, target_all = [], [], []
        # model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            features = self._network.feature_extractor(inputs)
            sim = []
            for i, task_project_f in enumerate(self._task_project_fs):
                if 'cos' in self._mode:
                    # cosine similarity
                    sim.append(task_project_f(features).mm(self._task_anchors[i].unsqueeze(-1)))
                elif 'euclidean' in self._mode:
                    # euclidean distance
                    sim.append(torch.norm(task_project_f(features)-self._task_anchors[i].unsqueeze(0), p=2, dim=1, keepdim=True))
                else:
                    raise ValueError('Unknown metric in mode: {}'.format(self._mode))
            
            sim = torch.cat(sim, dim=1)
            if 'cos' in self._mode:
                cnn_preds = torch.argmax(sim, dim=1) # for cosine similarity
            elif 'euclidean' in self._mode:
                cnn_preds = torch.argmin(sim, dim=1) # for euclidean distance
            else:
                raise ValueError('Unknown metric in mode: {}'.format(self._mode))
            
            
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
            else:
                if ret_task_acc:
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
                    cnn_task_correct += cnn_preds[task_data_idxs].eq(task_id).cpu().sum()
                    task_total += len(task_data_idxs)
                cnn_correct += cnn_preds.eq(task_id).cpu().sum()
                total += len(cnn_preds)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            return cnn_pred_all, [], np.ones_like(cnn_pred_all)*task_id
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                return test_acc, test_task_acc
            else:
                return test_acc
    
    def eval_task(self):        
        if len(self.cnn_task_metric_curve) == 0:
            self.cnn_task_metric_curve = np.zeros((self._nb_tasks, self._nb_tasks))

        self._logger.info(50*"-")
        self._logger.info("log {} of every task".format(self._eval_metric))
        self._logger.info(50*"-")
        if self._incre_type == 'cil':
            pass
        elif self._incre_type == 'til':
            cnn_pred, nme_pred, y_true = self.get_til_pred_target(self._network, self._test_dataset)

        if self._eval_metric == 'acc':
            cnn_total = np.around((cnn_pred==y_true).sum()*100/len(y_true), decimals=2)

            cnn_task = []
            for i in range(self._cur_task+1):
                idxes = np.where(y_true == i)[0]
                cnn_task.append(np.around((cnn_pred[idxes]==y_true[idxes]).sum()*100/len(idxes), decimals=2))

        elif self._eval_metric == 'mcr':
            pass
        else:
            raise ValueError('Unknown evaluate metric: {}'.format(self._eval_metric))
            
        self.cnn_metric_curve.append(cnn_total)
        self._logger.info("CNN : {} curve of all task is [\t".format(self._eval_metric) + 
            ("{:2.2f}\t"*len(self.cnn_metric_curve)).format(*self.cnn_metric_curve) + ']')
        for i in range(len(cnn_task)):
            self.cnn_task_metric_curve[i][self._cur_task] = cnn_task[i]
            self._logger.info("CNN : task {} {} curve is [\t".format(i, self._eval_metric)+
                        ("{:2.2f}\t"*len(cnn_task)).format(*self.cnn_task_metric_curve[i][:len(cnn_task)].tolist()) + ']')
        self._logger.info("CNN : Average (all task) {} of all stages: {:.2f}".format(self._eval_metric, np.mean(self.cnn_metric_curve)))
        self._logger.info("CNN : Average (last stage) {} of all task: {:.2f}".format(self._eval_metric, np.mean(self.cnn_task_metric_curve[:self._cur_task+1 ,self._cur_task])))
        self._logger.info("CNN : Backward Transfer: {:.2f}".format(cal_bwf(self.cnn_task_metric_curve, self._cur_task)))
        self._logger.info(' ')