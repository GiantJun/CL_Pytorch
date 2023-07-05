from os.path import join

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from backbone.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.replayBank import ReplayBank
from utils.toolkit import accuracy, count_parameters, tensor2numpy, cal_bwf, mean_class_recall, cal_class_avg_acc, cal_avg_forgetting, cal_openset_test_metrics

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
        # for openset test
        self.cnn_auc_curve = []
        self.nme_auc_curve = []
        self.cnn_fpr95_curve = []
        self.nme_fpr95_curve = []
        self.cnn_AP_curve = []
        self.nme_AP_curve = []

        self._incre_type = config.incre_type
        self._apply_nme = config.apply_nme
        self._memory_size = config.memory_size
        self._fixed_memory = config.fixed_memory
        self._sampling_method = config.sampling_method
        if self._fixed_memory:
            self._memory_per_class = config.memory_per_class
        self._memory_bank = None
        if self._fixed_memory != None: # memory replay only support cil or gem famalies
            self._memory_bank = ReplayBank(self._config, logger)
            self._logger.info('Memory bank created!')
        self._is_openset_test = config.openset_test
        
        self._replay_batch_size = config.batch_size if config.replay_batch_size is None else config.replay_batch_size

        self._init_epochs = config.epochs if config.init_epochs is None else config.init_epochs
        self._init_lrate = config.lrate if config.init_lrate is None else config.init_lrate
        self._init_scheduler = config.scheduler if config.init_scheduler is None else config.init_scheduler
        self._init_milestones = config.milestones if config.init_milestones is None else config.init_milestones
        self._init_lrate_decay = config.lrate_decay if config.init_lrate_decay is None else config.init_lrate_decay
        self._init_weight_decay = config.weight_decay if config.init_weight_decay is None else config.init_weight_decay
        self._init_opt_mom = config.opt_mom if config.init_opt_mom is None else config._init_opt_mom
                
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
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path, MLP_projector=self._config.MLP_projector)
        
        self._network.update_fc(self._total_classes)
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        if self._config.freeze_fe:
            self._network.freeze_FE()

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, feature_outputs = model(inputs)
            if self._incre_type == 'cil':
                loss = F.cross_entropy(logits[:,:task_end], targets)
                preds = torch.max(logits[:,:task_end], dim=1)[1]
            elif self._incre_type == 'til':
                loss = F.cross_entropy(logits[:, task_begin:task_end], targets-task_begin)
                preds = torch.max(logits[:, task_begin:task_end], dim=1)[1] + task_begin
    
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
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            if self._incre_type == 'cil':
                cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits[:,:task_end], dim=-1), dim=-1)
            elif self._incre_type == 'til':
                cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits[:, task_begin:task_end], dim=-1), dim=-1)
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
                if ret_task_acc:
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
                    cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]).cpu().sum()
                    task_total += len(task_data_idxs)
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
                test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                return test_acc, test_task_acc
            else:
                return test_acc

    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=epochs)
    
    def store_samples(self):
        if self._memory_bank != None:
            self._memory_bank.store_samples(self._sampler_dataset, self._network)
    
    def _train_model(self, model, train_loader, test_loader, optimizer, scheduler, task_id=None, epochs=100, note=''):
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        if note != '':
            note += '_'
        for epoch in range(epochs):
            model, train_acc, train_losses = self._epoch_train(model, train_loader, optimizer, scheduler,
                                task_begin=task_begin, task_end=task_end, task_id=task_id)
            # update record_dict
            record_dict = {}

            info = ('Task {}, Epoch {}/{} => '.format(task_id, epoch+1, epochs) + 
                ('{} {:.3f}, '* int(len(train_losses)/2)).format(*train_losses))
            for i in range(int(len(train_losses)/2)):
                record_dict['Task{}_{}'.format(task_id, note)+train_losses[i*2]] = train_losses[i*2+1]
            
            if train_acc is not None:
                record_dict['Task{}_{}Train_Acc'.format(task_id, note)] = train_acc
                info = info + 'Train_accy {:.2f}, '.format(train_acc)          
            
            test_acc, task_test_acc = self._epoch_test(model, test_loader, ret_task_acc=True,
                                                task_begin=task_begin, task_end=task_end, task_id=task_id)
            
            record_dict['Task{}_{}Test_Acc(inner task)'.format(task_id, note)] = task_test_acc
            info = info + 'Task{}_Test_accy {:.2f}, '.format(task_id, task_test_acc)
            if self._incre_type == 'cil': # only show epoch test acc in cil, because epoch test acc is worth nothing in til
                record_dict['Task{}_{}Test_Acc'.format(task_id, note)] = test_acc
                info = info + 'Test_accy {:.2f}'.format(test_acc)

            self._logger.info(info)
            self._logger.visual_log('train', record_dict, step=epoch)
        
        return model

    def eval_task(self):        
        if len(self.cnn_task_metric_curve) == 0:
            self.cnn_task_metric_curve = np.zeros((self._nb_tasks, self._nb_tasks))
            self.nme_task_metric_curve = np.zeros((self._nb_tasks, self._nb_tasks))

        self._logger.info(50*"-")
        self._logger.info("log {} of every task".format(self._eval_metric))
        self._logger.info(50*"-")
        if self._is_openset_test:
            test_dataset = self._openset_test_dataset
            test_loader = self._openset_test_loader
        else:
            test_dataset = self._test_dataset
            test_loader = self._test_loader
        if self._incre_type == 'cil':
            cnn_pred, nme_pred, cnn_pred_score, nme_pred_score, y_true, features = self.get_cil_pred_target(self._network, test_loader)
        elif self._incre_type == 'til':
            cnn_pred, nme_pred, cnn_pred_score, nme_pred_score, y_true, features = self.get_til_pred_target(self._network, test_dataset)
        
        # prepare for calculate openset auc
        if self._is_openset_test and self._cur_task < self._nb_tasks-1:
            openset_target = np.ones_like(y_true)
            openset_idx = np.where(y_true == sum(self._increment_steps))[0]
            openset_target[openset_idx] = 0
            cnn_openset_score = cnn_pred_score.copy()
            nme_openset_score = nme_pred_score.copy() if not (nme_pred is None or len(nme_pred) == 0) else None

            y_true = np.delete(y_true, openset_idx)
            cnn_pred = np.delete(cnn_pred, openset_idx)
            cnn_pred_score = np.delete(cnn_pred_score, openset_idx)
            nme_pred = np.delete(nme_pred, openset_idx) if not(nme_pred is None or len(nme_pred) == 0) else None
            nme_pred_score = np.delete(nme_pred_score, openset_idx) if not (nme_pred is None or len(nme_pred) == 0) else None
        
        # save predict and target records for more analysis afterwards
        if self._save_pred_record:
            self.save_predict_records(cnn_pred, cnn_pred_score, nme_pred_score, nme_pred, y_true, features)

        # start calculate and log out results(CNN)
        if self._eval_metric == 'acc':
            cnn_total, cnn_task = accuracy(cnn_pred.T, y_true, self._total_classes, self._increment_steps)
        elif self._eval_metric == 'mcr':
            cnn_total, cnn_task = mean_class_recall(cnn_pred.T, y_true, self._total_classes, self._increment_steps)
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
        self._logger.info("CNN : Final Average {} of all task: {:.2f}".format(self._eval_metric, np.mean(self.cnn_task_metric_curve[:self._cur_task+1 ,self._cur_task])))
        self._logger.info("CNN : Final Average ACC of all classes: {:.2f}".format(cal_class_avg_acc(cnn_pred, y_true)))
        self._logger.info("CNN : Backward Transfer: {:.2f}".format(cal_bwf(self.cnn_task_metric_curve, self._cur_task)))
        self._logger.info("CNN : Average Forgetting: {:.2f}".format(cal_avg_forgetting(self.cnn_task_metric_curve, self._cur_task)))
        # cal openset test metrics
        if self._is_openset_test and cnn_pred_score is not None:
            if self._cur_task < self._nb_tasks-1:
                roc_auc, fpr95, ap = cal_openset_test_metrics(cnn_openset_score, openset_target)
                self.cnn_auc_curve.append(roc_auc)
                self.cnn_fpr95_curve.append(fpr95)
                self.cnn_AP_curve.append(ap)
            self._logger.info("CNN : openset AUC curve of all stages is [\t" + ("{:2.2f}\t"*len(self.cnn_auc_curve)).format(*self.cnn_auc_curve) + ']')
            self._logger.info("CNN : Average AUC of all stages: {:.2f}".format(np.mean(self.cnn_auc_curve)))
            self._logger.info("CNN : openset fpr95 curve of all stages is [\t" + ("{:2.2f}\t"*len(self.cnn_fpr95_curve)).format(*self.cnn_fpr95_curve) + ']')
            self._logger.info("CNN : Average fpr95 of all stages: {:.2f}".format(np.mean(self.cnn_fpr95_curve)))
            self._logger.info("CNN : openset AP curve of all stages is [\t" + ("{:2.2f}\t"*len(self.cnn_AP_curve)).format(*self.cnn_AP_curve) + ']')
            self._logger.info("CNN : Average AP of all stages: {:.2f}".format(np.mean(self.cnn_AP_curve)))
        self._logger.info(' ')
    
        # start calculate and log out results(NME)
        if not (nme_pred is None or len(nme_pred) == 0):
            if self._eval_metric == 'acc':
                nme_total, nme_task = accuracy(nme_pred.T, y_true, self._total_classes, self._increment_steps)
            elif self._eval_metric == 'mcr':
                nme_total, nme_task = mean_class_recall(nme_pred.T, y_true, self._total_classes, self._increment_steps)
            else:
                raise ValueError('Unknown evaluate metric: {}'.format(self._eval_metric))
            self.nme_metric_curve.append(nme_total)
            self._logger.info("NME : {} curve of all task is [\t".format(self._eval_metric) +
                ("{:2.2f}\t"*len(self.nme_metric_curve)).format(*self.nme_metric_curve) + ']')
            for i in range(len(nme_task)):
                self.nme_task_metric_curve[i][self._cur_task] = nme_task[i]
                self._logger.info("NME : task {} {} curve is [\t".format(i, self._eval_metric) + 
                        ("{:2.2f}\t"*len(nme_task)).format(*self.nme_task_metric_curve[i][:len(nme_task)].tolist()) + ']')
            self._logger.info("NME : Average (all task) {} of all stages: {:.2f}".format(self._eval_metric, np.mean(self.nme_metric_curve)))
            self._logger.info("NME : Average (last stage) {} of all task: {:.2f}".format(self._eval_metric, np.mean(self.nme_task_metric_curve[:self._cur_task+1 ,self._cur_task])))
            self._logger.info("NME : Average ACC of all classes: {:.2f}".format(cal_class_avg_acc(nme_pred, y_true)))
            self._logger.info("NME : Backward Transfer: {:.2f}".format(cal_bwf(self.nme_task_metric_curve, self._cur_task)))
            self._logger.info("NME : Average Forgetting: {:.2f}".format(cal_avg_forgetting(self.nme_task_metric_curve, self._cur_task)))
            # cal openset test metrics
            if self._is_openset_test and nme_pred_score is not None:
                if self._cur_task < self._nb_tasks-1:
                    roc_auc, fpr95, ap = cal_openset_test_metrics(nme_openset_score, openset_target)
                    self.nme_auc_curve.append(roc_auc)
                    self.nme_fpr95_curve.append(fpr95)
                    self.nme_AP_curve.append(ap)
                self._logger.info("NME : openset AUC curve of all stages is [\t" + ("{:2.2f}\t"*len(self.nme_auc_curve)).format(*self.nme_auc_curve) + ']')
                self._logger.info("NME : Average AUC of all stages: {:.2f}".format(np.mean(self.nme_auc_curve)))
                self._logger.info("NME : openset fpr95 curve of all stages is [\t" + ("{:2.2f}\t"*len(self.nme_fpr95_curve)).format(*self.nme_fpr95_curve) + ']')
                self._logger.info("NME : Average fpr95 of all stages: {:.2f}".format(np.mean(self.nme_fpr95_curve)))
                self._logger.info("NME : openset AP curve of all stages is [\t" + ("{:2.2f}\t"*len(self.nme_AP_curve)).format(*self.nme_AP_curve) + ']')
                self._logger.info("NME : Average AP of all stages: {:.2f}".format(np.mean(self.nme_AP_curve)))
            self._logger.info(' ')

    def after_task(self):
        self._known_classes = self._total_classes
        if self._save_models:
            self.save_checkpoint('seed{}_task{}_checkpoint.pkl'.format(self._seed, self._cur_task),
                self._network.cpu(), self._cur_task)

    def save_checkpoint(self, filename, model, task_id):
        save_path = join(self._logdir, filename)
        if self._memory_bank is None:
            memory_class_means = None
        else:
            memory_class_means = self._memory_bank.get_class_means()
        if isinstance(model, nn.DataParallel):
            model = model.module
        save_dict = {'state_dict': model.state_dict(), 'config':self._config.get_parameters_dict(),
                'task_id': task_id, 'memory_class_means':memory_class_means}
        torch.save(save_dict, save_path)
        self._logger.info('checkpoint saved at: {}'.format(save_path))
    
    def save_predict_records(self, cnn_pred, cnn_pred_scores, nme_pred, nme_pred_scores, targets, features):
        record_dict = {}        
        record_dict['cnn_pred'] = cnn_pred
        record_dict['cnn_pred_scores'] = cnn_pred_scores
        record_dict['nme_pred'] = nme_pred
        record_dict['nme_pred_scores'] = nme_pred_scores
        record_dict['targets'] = targets
        record_dict['features'] = features

        filename = 'pred_record_seed{}_task{}.npy'.format(self._seed, self._cur_task)
        np.save(join(self._logdir, filename), record_dict)
    
    def get_cil_pred_target(self, model, test_loader):
        return self._epoch_test(model, test_loader, ret_pred_target=True, task_begin=0, 
                    task_end=self._total_classes, task_id=self._cur_task)
    
    def get_til_pred_target(self, model, test_dataset):
        known_classes = 0
        total_classes = 0
        cnn_pred_result, nme_pred_result, y_true_result, cnn_predict_score, nme_predict_score, features_result = [], [], [], [], [], []
        for task_id in range(self._cur_task + 1):
            cur_classes = self._increment_steps[task_id]
            total_classes += cur_classes
            
            task_dataset = Subset(test_dataset, 
                np.argwhere(np.logical_and(test_dataset.targets >= known_classes, test_dataset.targets < total_classes)).squeeze())
            task_loader = DataLoader(task_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            
            cnn_pred, nme_pred, cnn_pred_score, nme_pred_score, y_true, features = self._epoch_test(model, task_loader,
                        ret_pred_target=True, task_begin=known_classes, task_end=total_classes, task_id=task_id)
            cnn_pred_result.append(cnn_pred)
            y_true_result.append(y_true)
            cnn_predict_score.append(cnn_pred_score)
            features_result.append(features)

            known_classes = total_classes

        cnn_pred_result = np.concatenate(cnn_pred_result)
        y_true_result = np.concatenate(y_true_result)
        cnn_predict_score = np.concatenate(cnn_predict_score)
        features_result = np.concatenate(features_result)

        return cnn_pred_result, nme_pred_result, cnn_predict_score, nme_predict_score, y_true_result, features_result
    
    def release(self):
        super().release()
        if self._memory_bank is not None:
            self._memory_bank = None

    def _get_optimizer(self, params, config, is_init:bool):
        optimizer = None
        if is_init:
            if config.opt_type == 'sgd':
                optimizer = optim.SGD(params, lr=self._init_lrate,
                                      momentum=0 if self._init_opt_mom is None else self._init_opt_mom,
                                      weight_decay=0 if self._init_weight_decay is None else self._init_weight_decay)
                self._logger.info('Applying sgd: lr={}, momenton={}, weight_decay={}'.format(self._init_lrate, self._init_opt_mom, self._init_weight_decay))
            elif config.opt_type == 'adam':
                optimizer = optim.Adam(params, lr=self._init_lrate,
                                       weight_decay=0 if self._init_weight_decay is None else self._init_weight_decay)
                self._logger.info('Applying adam: lr={}, weight_decay={}'.format(self._init_lrate, self._init_weight_decay))
            elif config.opt_type == 'adamw':
                optimizer = optim.AdamW(params, lr=self._init_lrate,
                                        weight_decay=0 if self._init_weight_decay is None else self._init_weight_decay,)
                self._logger.info('Applying adamw: lr={}, weight_decay={}'.format(self._init_lrate, self._init_weight_decay))
            else:
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        else:
            if config.opt_type == 'sgd':
                optimizer = optim.SGD(params, lr=config.lrate,
                                      momentum=0 if config.opt_mom is None else config.opt_mom,
                                      weight_decay=0 if config.weight_decay is None else config.weight_decay)
                self._logger.info('Applying sgd: lr={}, momenton={}, weight_decay={}'.format(config.lrate, config.opt_mom, config.weight_decay))
            elif config.opt_type == 'adam':
                optimizer = optim.Adam(params, lr=config.lrate,
                                       weight_decay=0 if config.weight_decay is None else config.weight_decay)
                self._logger.info('Applying adam: lr={}, weight_decay={}'.format(config.lrate, config.weight_decay))
            else: 
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        return optimizer
    
    def _get_scheduler(self, optimizer, config, is_init:bool):
        scheduler = None
        if is_init:
            if config.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self._init_milestones, gamma=self._init_lrate_decay)
                self._logger.info('Applying multi_step scheduler: lr_decay={}, milestone={}'.format(self._init_lrate_decay, self._init_milestones))
            elif config.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self._init_epochs)
                self._logger.info('Applying cos scheduler')
            elif config.scheduler == None:
                scheduler = None
            else: 
                raise ValueError('Unknown scheduler: {}'.format(config.scheduler))
        else:
            if config.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.milestones, gamma=config.lrate_decay)
                self._logger.info('Applying multi_step scheduler: lr_decay={}, milestone={}'.format(config.lrate_decay, config.milestones))
            elif config.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs)
                self._logger.info('Applying cos scheduler: T_max={}'.format(config.epochs))
            # elif config.scheduler == 'coslrs':
            #     scheduler = optim.CosineLRScheduler(optimizer, t_initial=self._init_epochs, decay_rate=0.1, lr_min=1e-5, warmup_t=5, warmup_lr_init=1e-6)
            elif config.scheduler == None:
                scheduler = None
            else: 
                raise ValueError('No scheduler: {}'.format(config.scheduler))
        return scheduler