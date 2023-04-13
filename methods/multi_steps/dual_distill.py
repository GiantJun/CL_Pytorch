import numpy as np
import torch
import copy
from torch.nn.functional import cross_entropy

from torch.utils.data import DataLoader
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import target2onehot, tensor2numpy
from backbone.inc_net import IncrementalNet


EPSILON = 1e-8

class Dual_Distill(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._origin_network = None
        self._expert_network = None
        self._T = config.T
        self._Tn = config.T
        self._lamda1 = config.alpha
        self._lamda2 = config.beta
        self._is_stage_two = False
        self._class_mean_all = {} # key is task id
        if self._incre_type != 'cil':
            raise ValueError('dual_distill is a class incremental method!')
    
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
        
        self._new_task_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._new_task_loader = DataLoader(self._new_task_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')
    
    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._cur_task > 0:
            self._expert_network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
            self._expert_network.feature_extractor.load_state_dict(self._origin_network.feature_extractor.state_dict())
            self._expert_network.update_fc(self._total_classes - self._known_classes)
            self._expert_network = self._expert_network.cuda()
            
        if self._origin_network is not None:
            self._origin_network.cuda()
    
    def after_task(self):
        super().after_task()
        self._origin_network = self._network.copy().freeze()

    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        if self._cur_task == 0:
            self._is_stage_two = False
            optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
            scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
            self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=self._init_epochs)
        else:
            self._is_stage_two = False
            self._logger.info('Training expert model...')
            optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._expert_network.parameters()), self._config, self._cur_task==0)
            scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
            self._expert_network = self._train_model(self._expert_network, self._new_task_loader, self._test_loader, optimizer, scheduler,
                task_id=self._cur_task, epochs=self._epochs, note='stage1')
            self._expert_network.eval()
            self._expert_network.freeze()

            self._is_stage_two = True
            self._logger.info('Training updated model...')
            optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
            scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
            self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
                task_id=self._cur_task, epochs=self._init_epochs, note='stage2')

    def store_samples(self):
        if self._cur_task > 0:
            old_task_class_means = self._memory_bank.get_class_means()
            new_task_class_means = self._memory_bank.cal_class_means(self._network, self._sampler_dataset)
            origin_class_means = torch.cat([old_task_class_means, new_task_class_means], dim=0)
            
            self._memory_bank.store_samples(self._sampler_dataset, self._network)
            updated_class_means = self._memory_bank.get_class_means()

            known_class_num, total_class_num = 0, 0
            emsemble_class_means = []
            for task_id, cur_class_num in enumerate(self._increment_steps[:self._cur_task+1]):
                total_class_num += cur_class_num

                origin_task_class_means = origin_class_means[known_class_num: total_class_num]
                updated_task_class_means = updated_class_means[known_class_num: total_class_num]

                task_weight = self._cur_task - task_id
                emsambled_task_class_means = (task_weight*origin_task_class_means + updated_task_class_means) / (task_weight + 1)
                emsemble_class_means.append(emsambled_task_class_means)

                known_class_num = total_class_num

            emsemble_class_means = torch.cat(emsemble_class_means, dim=0)
            self._memory_bank.set_class_means(emsemble_class_means)
        else:
            self._memory_bank.store_samples(self._sampler_dataset, self._network)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, origin_kd_losses, expert_kd_losses = 0., 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)

            if task_id == 0:
                ce_loss = cross_entropy(logits, targets)
                ce_losses += ce_loss.item()
                loss = ce_loss
            elif not self._is_stage_two: # training expert network
                targets -= task_begin
                ce_loss = cross_entropy(logits, targets)
                ce_losses += ce_loss.item()
                loss = ce_loss
            else: # training updated network
                ce_loss = cross_entropy(logits[:,:task_end], targets)
                ce_losses += ce_loss.item()

                origin_kd_loss = self._KD_loss(logits[:,:task_begin], self._origin_network(inputs)[0], self._T)
                origin_kd_losses += origin_kd_loss.item()

                expert_kd_loss = self._KD_loss(logits[:, task_begin:], self._expert_network(inputs)[0], self._Tn)
                expert_kd_losses += expert_kd_loss.item()
                
                loss = ce_loss + self._lamda1*origin_kd_loss + self._lamda2*expert_kd_loss

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
        if self._is_stage_two:
            train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader), 'Loss_origin_kd', origin_kd_losses/len(train_loader),
                'Loss_expert_kd', expert_kd_losses/len(train_loader)]
        else:
            train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader)]
        return model, train_acc, train_loss

    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        cnn_pred_all, nme_pred_all, target_all = [], [], []
        cnn_max_scores_all, nme_max_scores_all = [], []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits[:,:task_end], dim=-1), dim=-1)
                
            if ret_pred_target:
                if self._memory_bank != None and self._apply_nme:
                    nme_pred, nme_max_scores = self._memory_bank.KNN_classify(feature_outputs['features'])
                    nme_pred_all.append(tensor2numpy(nme_pred))
                    nme_max_scores_all.append(tensor2numpy(nme_max_scores))
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
            else:
                if ret_task_acc:
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
                    if task_id > 0 and not self._is_stage_two:
                        cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]-task_begin).cpu().sum()
                    else:
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
            return cnn_pred_all, nme_pred_all, cnn_max_scores_all, nme_max_scores_all, target_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                return test_acc, test_task_acc
            else:
                return test_acc

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]