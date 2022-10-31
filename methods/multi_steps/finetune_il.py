from os.path import join

import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Subset

from backbone.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.replayBank import ReplayBank
from utils.toolkit import accuracy, count_parameters, tensor2numpy, cal_bwf

EPSILON = 1e-8

'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

# base is finetune with or without memory_bank
class Finetune_IL(BaseLearner):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        # 评价指标变化曲线
        self.cnn_task_metric_curve = []
        self.nme_task_metric_curve = []
        self.cnn_metric_curve = []
        self.nme_metric_curve = []

        self._incre_type = config.incre_type
        self._apply_nme = config.apply_nme
        self._memory_size = config.memory_size
        self._fixed_memory = config.fixed_memory
        self._sampling_method = config.sampling_method
        if self._fixed_memory:
            self._memory_per_class = config.memory_per_class
        self._memory_bank = None
        if (self._memory_size != None and self._fixed_memory != None and 
            self._sampling_method != None and self._incre_type == 'cil'):
            self._memory_bank = ReplayBank(self._config)
            self._logger.info('Memory bank created!')
        
        self._init_epochs = config.epochs if config.init_epochs == None else config.init_epochs
        self._init_lrate = config.lrate if config.init_lrate == None else config.init_lrate
        self._init_scheduler = config.scheduler if config.init_scheduler == None else config.init_scheduler
        self._init_milestones = config.milestones if config.init_milestones == None else config.init_milestones
        self._init_lrate_decay = config.lrate_decay if config.init_lrate_decay == None else config.init_lrate_decay
        self._init_weight_decay = config.weight_decay if config.init_weight_decay == None else config.init_weight_decay
                
    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        if self._cur_task > 0 and self._memory_bank != None:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), 
                    source='train', mode='train', appendent=self._memory_bank.get_memory())
        else:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')

    def prepare_model(self):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path).cuda()
        if self._incre_type == 'cil':
            self._network.update_fc(self._total_classes)
        elif self._incre_type == 'til':
            self._network.update_til_fc(self._cur_classes)
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

    def eval_task(self):        
        if len(self.cnn_task_metric_curve) == 0:
            self.cnn_task_metric_curve = np.zeros((self._nb_tasks, self._nb_tasks))
            self.nme_task_metric_curve = np.zeros((self._nb_tasks, self._nb_tasks))

        self._logger.info(50*"-")
        self._logger.info("log {} of every task".format(self._eval_metric))
        self._logger.info(50*"-")
        if self._incre_type == 'cil':
            cnn_pred, nme_pred, y_true = self.get_cil_pred_target(self._network, self._test_loader)
        elif self._incre_type == 'til':
            cnn_pred, nme_pred, y_true = self.get_til_pred_target(self._network, self._test_dataset)

        # 计算top1, 这里去掉了计算 topk 的代码
        if self._eval_metric == 'acc':
            cnn_total, cnn_task = accuracy(cnn_pred.T, y_true, self._total_classes, self._increment_steps)
        else:
            pass
        self.cnn_metric_curve.append(cnn_total)
        self._logger.info("CNN : {} curve of all task is [\t".format(self._eval_metric) + 
            ("{:2.2f}\t"*len(self.cnn_metric_curve)).format(*self.cnn_metric_curve) + ']')
        for i in range(len(cnn_task)):
            self.cnn_task_metric_curve[i][self._cur_task] = cnn_task[i]
            self._logger.info("CNN : task {} {} curve is [\t".format(i, self._eval_metric)+
                        ("{:2.2f}\t"*len(self.cnn_task_metric_curve[i])).format(*self.cnn_task_metric_curve[i].tolist()) + ']')
        self._logger.info("CNN : Average Incremental Acc: {:.2f}".format(np.mean(self.cnn_metric_curve)))
        self._logger.info("CNN : Backward Transfer: {:.2f}".format(cal_bwf(self.cnn_task_metric_curve, self._cur_task)))
        self._logger.info(' ')
    
        if len(nme_pred) != 0:
            if self._eval_metric == 'acc':
                nme_total, nme_task = accuracy(nme_pred.T, y_true, self._total_classes, self._increment_steps)
            else:
                pass
            self.nme_metric_curve.append(nme_total)
            self._logger.info("NME : {} curve of all task is [\t".format(self._eval_metric) +
                ("{:2.2f}\t"*len(self.nme_metric_curve)).format(*self.nme_metric_curve) + ']')
            for i in range(len(nme_task)):
                self.nme_task_metric_curve[i][self._cur_task] = nme_task[i]
                self._logger.info("NME : task {} {} curve is [\t".format(i, self._eval_metric) + 
                        ("{:2.2f}\t"*len(self.nme_task_metric_curve[i])).format(*self.nme_task_metric_curve[i].tolist()) + ']')
            self._logger.info("NME : Average Incremental Acc: {:2.2f}".format(np.mean(self.nme_metric_curve)))
            self._logger.info("NME : Backward Transfer: {:.2f}".format(cal_bwf(self.nme_task_metric_curve, self._cur_task)))
            self._logger.info(' ')

    # need to be overwrite probably
    def after_task(self):
        self._known_classes = self._total_classes
        if self._save_models:
            self.save_checkpoint('{}_{}_{}_task{}_seed{}.pkl'.format(
                self._method, self._dataset, self._backbone, self._cur_task, self._seed), 
                self._network.cpu(), self._cur_task)
        self._logger.info(' ')

    def save_checkpoint(self, filename, model, task_id):
        save_path = join(self._logdir, filename)
        save_dict = {'model': model, 'config':self._config.get_save_config(), 'task_id': task_id}
        torch.save(save_dict, save_path)
        self._logger.info('model saved at: {}'.format(save_path))
    
    def get_cil_pred_target(self, model, test_loader):
        return self._epoch_test(model, test_loader, True)
    
    def get_til_pred_target(self, model, test_dataset):
        known_classes = 0
        total_classes = 0
        cnn_pred_result, nme_pred_result, y_true_result = [], [], []
        for task_id in range(self._cur_task + 1):
            cur_classes = self._increment_steps[task_id]
            total_classes += cur_classes
            
            task_dataset = Subset(test_dataset, 
                np.argwhere(np.logical_and(test_dataset.targets >= known_classes, test_dataset.targets < total_classes + 1)))
            task_loader = DataLoader(task_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            
            cnn_pred, nme_pred, y_true = self._epoch_test(model, task_loader, True)
            cnn_pred_result.append(cnn_pred+known_classes)
            if len(nme_pred) > 0:
                nme_pred_result.append(nme_pred+known_classes)
            y_true_result.append(y_true)
            
            known_classes = total_classes

        if len(nme_pred_result) == 1:
            cnn_pred_result = cnn_pred_result[0]
            nme_pred_result = nme_pred_result[0]
            y_true_result = y_true_result[0]
        elif len(nme_pred_result) > 1:
            cnn_pred_result = np.concatenate(cnn_pred_result)
            nme_pred_result = np.concatenate(nme_pred_result)
            y_true_result = np.concatenate(y_true_result)
        else:
            cnn_pred_result = np.concatenate(cnn_pred_result)
            y_true_result = np.concatenate(y_true_result)

        return cnn_pred_result, nme_pred_result, y_true_result
    
    def _get_optimizer(self, params, config, is_init:bool):
        optimizer = None
        if is_init:
            if config.opt_type == 'sgd':
                optimizer = optim.SGD(params, momentum=0.9, lr=self._init_lrate, weight_decay=self._init_weight_decay)
            elif config.opt_type == 'adam':
                optimizer = optim.Adam(params, lr=self._init_lrate)
            else: 
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        else:
            if config.opt_type == 'sgd':
                optimizer = optim.SGD(params, momentum=0.9, lr=config.lrate, weight_decay=config.weight_decay)
            elif config.opt_type == 'adam':
                optimizer = optim.Adam(params, lr=config.lrate)
            else: 
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        return optimizer
    
    def _get_scheduler(self, optimizer, config, is_init:bool):
        scheduler = None
        if is_init:
            if config.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self._init_milestones, gamma=self._init_lrate_decay)
            elif config.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self._init_epochs)
            else: 
                raise ValueError('No scheduler: {}'.format(config.scheduler))
        else:
            if config.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.milestones, gamma=config.lrate_decay)
            elif config.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs)
            else: 
                raise ValueError('No scheduler: {}'.format(config.scheduler))
        return scheduler