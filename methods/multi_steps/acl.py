import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from argparse import ArgumentParser

from backbone.adapter_cl_net import CNN_Adapter_Net_CIL_V2
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy


EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--layer_names', nargs='+', type=str, default=None, help='layers to apply prompt, e.t. [layer1, layer2]')
    parser.add_argument('--epochs_finetune', type=int, default=None, help='balance finetune epochs')
    parser.add_argument('--lrate_finetune', type=float, default=None, help='balance finetune learning rate')
    parser.add_argument('--milestones_finetune', nargs='+', type=int, default=None, help='for multi step learning rate decay scheduler')
    return parser

class ACL(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._layer_names = config.layer_names

        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._milestones_finetune = config.milestones_finetune

        self._is_training_adapters = False

        if self._incre_type != 'cil':
            raise ValueError('Continual_Adapter is a class incremental method!')

        self._class_means_list = []

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        if self._cur_task > 0:
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
            self._network = CNN_Adapter_Net_CIL_V2(self._logger, self._config.backbone, self._config.pretrained,
                    pretrain_path=self._config.pretrain_path, layer_names=self._layer_names)
        self._network.update_fc(self._total_classes)

        self._network.freeze_FE()
        self._network = self._network.cuda()

        self._logger.info('Initializing task-{} adapter in network...'.format(self._cur_task))
        with torch.no_grad():
            self._network.eval()
            self._network.train_adapter_mode()
            self._network(torch.rand(1, 3, self._config.img_size, self._config.img_size).cuda())
        self._network.freeze_adapters(mode='old')

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        all_params, all_trainable_params = 0, 0
        for layer_id in self._config.layer_names:
            adapter_id = layer_id.replace('.', '_')+'_adapters'
            if hasattr(self._network, adapter_id):
                adapter_module = getattr(self._network, adapter_id)
                
                layer_params = count_parameters(adapter_module)
                layer_trainable_params = count_parameters(adapter_module, True)
                self._logger.info('{} params: {} , trainable params: {}'.format(adapter_id,
                    layer_params, layer_trainable_params))
                
                all_params += layer_params
                all_trainable_params += layer_trainable_params
        self._logger.info('all adapters params: {} , trainable params: {}'.format(all_params, all_trainable_params))
        self._logger.info('seperate fc params: {} , trainable params: {}'.format(count_parameters(self._network.seperate_fc), count_parameters(self._network.seperate_fc, True)))
        self._logger.info('aux fc params: {} , trainable params: {}'.format(count_parameters(self._network.aux_fc), count_parameters(self._network.aux_fc, True)))
        
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} requre grad!'.format(name))

    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        
        # train adapters
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, False)
        scheduler = self._get_scheduler(optimizer, self._config, False)
        self._is_training_adapters = True
        self._network.train_adapter_mode()
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
            task_id=self._cur_task, epochs=self._epochs, note='stage1')

        if self._cur_task == 0:
            self._network.seperate_fc[0].load_state_dict(self._network.aux_fc.state_dict())
            self._is_training_adapters = False
        else:
            ### stage 2: Retrain fc (begin) ###
            self._logger.info('Retraining the network (adapter feature fusion part) with the balanced dataset!')
            finetune_train_dataset = self._memory_bank.get_unified_sample_dataset(self._train_dataset, self._network)
            finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self._batch_size,
                                            shuffle=True, num_workers=self._num_workers)

            self._is_training_adapters = False
            self._network.test_mode()
            self._network.freeze_adapters(mode='all')

            ft_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self._lrate_finetune)
            ft_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=ft_optimizer, milestones=self._milestones_finetune, gamma=self._config.lrate_decay)
            
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    self._logger.info('{} requre grad!'.format(name))

            self._network = self._train_model(self._network, finetune_train_loader, self._test_loader, ft_optimizer, ft_scheduler, 
                task_id=self._cur_task, epochs=self._epochs_finetune, note='stage2')
            ### stage 2: Retrain fc (end) ###
    
    def store_samples(self):
        if self._memory_bank is not None:
            self._network.store_sample_mode()
            self._memory_bank.store_samples(self._sampler_dataset, self._network)
        # prepare for eval()
        self._network.test_mode()

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        correct, total = 0, 0
        ce_losses = 0.
        losses = 0.
        
        if self._is_training_adapters:
            model.new_adapters_train()
        else:
            model.eval()

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            # model forward has two mode which shoule be noticed before forward!
            logits, output_features = model(inputs)
            
            if self._is_training_adapters:
                targets = torch.where(targets-task_begin+1>0, targets, task_end)
                loss = F.cross_entropy(logits, targets-task_begin)
                preds = torch.argmax(logits, dim=1) + task_begin
                correct += preds.eq(targets).cpu().sum()
            else:
                # stage 2
                loss = torch.tensor(0.0)
                known_class_num, total_class_num = 0, 0
                for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]):
                    total_class_num += cur_class_num + 1
                    task_logits = logits[:,known_class_num:total_class_num]
                    
                    task_targets = (torch.ones(targets.shape[0], dtype=int) * cur_class_num).cuda() # class label: [0, cur_class_num]
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=known_class_num-id, targets<total_class_num-id-1)).squeeze(-1)
                    if len(task_data_idxs) > 0:
                        task_targets[task_data_idxs] = targets[task_data_idxs] - known_class_num + id
                        loss = loss + F.cross_entropy(task_logits, task_targets)

                    if id == task_id:
                        preds = torch.argmax(logits[:,known_class_num:total_class_num], dim=1)

                    known_class_num = total_class_num
                
                ce_losses += loss.item()

                aux_targets = torch.where(targets-task_begin+1>0, targets-task_begin, task_end - task_begin)
                correct += preds.eq(aux_targets).cpu().sum()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        
        # train acc shows the performance of the model on current task
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'ce_loss', ce_losses/len(train_loader)]
        return model, train_acc, train_loss
    
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        task_id_correct = 0
        cnn_pred_all, target_all = [], []
        cnn_max_scores_all = []
        features_all = []
        model.eval()

        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            # model forward has two mode which shoule be noticed before forward!
            logits, feature_outputs = model(inputs)

            if self._is_training_adapters:
                targets = torch.where(targets-task_begin+1>0, targets, task_end)
                cnn_preds = torch.argmax(logits, dim=1) + task_begin
            else:
                # task_test_acc 有意义(反映当前task的预测准确率), test_acc 有意义(反映模型最终的预测结果)
                ### predict task id based on the unknown scores ###
                cnn_preds, cnn_max_scores, task_id_pred_correct= self.min_others_test(logits=logits, targets=targets, task_id=task_id)
                task_id_correct += task_id_pred_correct

            if ret_pred_target: # only apply when self._is_training_adapters is False
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
                features_all.append(tensor2numpy(feature_outputs['features']))
            else:
                if ret_task_acc:
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
                    cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]).cpu().sum()
                    task_total += len(task_data_idxs)
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
            
            # for print out task id predict acc
            total += len(targets)
        
        if not self._is_training_adapters:
            self._logger.info('Test task id predict acc (CNN) : {:.2f}'.format(np.around(task_id_correct*100 / total, decimals=2)))

        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            features_all = np.concatenate(features_all)
            return cnn_pred_all, None, cnn_max_scores_all, None, target_all, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                return test_acc, test_task_acc
            else:
                return test_acc
    
    def min_others_test(self, logits, targets, task_id):
         ### predict task id based on the unknown scores ###
        unknown_scores = []
        known_scores = []
        known_class_num, total_class_num = 0, 0
        task_id_targets = torch.zeros(targets.shape[0], dtype=int).cuda()
        for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]):
            total_class_num += cur_class_num + 1
            task_logits = logits[:,known_class_num:total_class_num]
            task_scores = torch.softmax(task_logits, dim=1)

            unknown_scores.append(task_scores[:, -1])
            
            known_task_scores = torch.zeros((targets.shape[0], max(self._increment_steps))).cuda()
            known_task_scores[:, :(task_scores.shape[1]-1)] = task_scores[:, :-1]
            known_scores.append(known_task_scores)

            # generate task_id_targets
            task_data_idxs = torch.argwhere(torch.logical_and(targets>=known_class_num-id, targets<total_class_num-id-1)).squeeze(-1)
            if len(task_data_idxs) > 0:
                task_id_targets[task_data_idxs] = id

            known_class_num = total_class_num

        known_scores = torch.stack(known_scores, dim=0) # task num, b, max(task_sizes)
        unknown_scores = torch.stack(unknown_scores, dim=-1) # b, task num
    
        min_scores, task_id_predict = torch.min(unknown_scores, dim=-1)
        cnn_max_scores = 1 - min_scores
        ###
        
        ### predict class based on task id and known scores ###
        cnn_preds = torch.zeros(targets.shape[0], dtype=int).cuda()
        known_class_num, total_class_num = 0, 0
        for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]):
            total_class_num += cur_class_num # do not have others category !
            task_logits_idxs = torch.argwhere(task_id_predict==id).squeeze(-1)
            if len(task_logits_idxs) > 0:
                cnn_preds[task_logits_idxs] = torch.argmax(known_scores[id, task_logits_idxs], dim=1) + known_class_num
                
            known_class_num = total_class_num
        
        task_id_correct = task_id_predict.eq(task_id_targets).cpu().sum()

        return cnn_preds, cnn_max_scores, task_id_correct