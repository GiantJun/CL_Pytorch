import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone.cnn_prompt_net import ProtoNet
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters, tensor2numpy

EPSILON = 1e-8


class CNN_Prompt_v2(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = self._config.mode.split('|')
        self._nb_proxy = self._config.nb_proxy
        self._gamma = self._config.gamma
 
    def prepare_task_data(self, data_manager):
        self._cur_task = data_manager.nb_tasks - 1
        self._cur_classes = data_manager.get_task_size(0)
        self._total_classes = self._known_classes + self._cur_classes
        train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='train')
        test_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(test_dataset)))

        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)


    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = ProtoNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path,
                    nb_proxy=self._nb_proxy, dist_mode=self._config.mode, use_MLP=self._config.use_MLP)
        self._network.update_fc(self._total_classes)
        if self._checkpoint is not None:
            self._network.load_state_dict(self._checkpoint['state_dict'])
        
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
        ce_losses, proxy_distinc_losses = 0., 0.
        correct, total = 0, 0
        model.eval()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            if 'pull_far_cluster' in self._mode or 'pull_close_cluster' in self._mode:
                logits, feature_outputs = model(inputs, targets)
            else:
                logits, feature_outputs = model(inputs)
        
            if 'cosine' in self._mode:
                ce_loss = F.cross_entropy(logits, targets)
                preds = torch.max(logits, dim=1)[1]
            else: # euclidean
                ce_loss = F.cross_entropy(-1 * logits, targets)
                preds = torch.min(logits, dim=1)[1]

            proxy_distinc_loss = torch.tensor([0.]).cuda()
            if 'distinc_proxy' in self._mode:
                proxy_distinc_loss = model.get_proxy_distinc_loss()
                if 'euclidean' in self._mode:
                    proxy_distinc_loss *= -1

            loss = ce_loss + self._gamma * proxy_distinc_loss
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            ce_losses += ce_loss.item()
            proxy_distinc_losses += proxy_distinc_loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'CE_loss', ce_losses/len(train_loader), 'proxy_distinc_loss', proxy_distinc_losses/len(train_loader)]
        return model, train_acc, train_loss
    
    def _epoch_test(self, model, test_loader, ret_pred_target=False):
        cnn_correct, total = 0, 0
        cnn_pred_all, target_all = [], []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feature_outputs = model(inputs)
            if 'cosine' in self._mode:
                cnn_preds = torch.max(outputs, dim=1)[1]
            else: # euclidean
                cnn_preds = torch.min(outputs, dim=1)[1]
            
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
            else:
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            return cnn_pred_all, target_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc