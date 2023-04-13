import numpy as np
import torch
from torch.nn.functional import cross_entropy

from backbone.cnn_adapter_net import CNN_Task_Adapter_Net
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8


class CNN_Task_Adapter(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = self._config.mode.split('|')

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = CNN_Task_Adapter_Net(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path,
                    layer_names=self._config.layer_names, mode=self._mode)
            self._logger.info('Created Task-specific prompt net!')
        self._network.update_fc(self._total_classes)
        if self._config.freeze_fe:
            self._network.freeze_FE()
        self._network = self._network.cuda()
        
        self._logger.info('Initializing adapter in network...')
        with torch.no_grad():
            self._network.eval()
            self._network(torch.rand(1, 3, self._config.img_size, self._config.img_size).cuda())

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        all_params, all_trainable_params = 0, 0
        for layer_id in self._config.layer_names:
            if hasattr(self._network, layer_id+'_adapter'):
                adapter_module = getattr(self._network, layer_id+'_adapter')
                
                layer_params = count_parameters(adapter_module)
                layer_trainable_params = count_parameters(adapter_module, True)
                self._logger.info('{} params: {} , trainable params: {}'.format(layer_id+'_adapter', 
                    layer_params, layer_trainable_params))
                
                all_params += layer_params
                all_trainable_params += layer_trainable_params
            
        self._logger.info('all adapters params: {} , trainable params: {}'.format(all_params, all_trainable_params))

        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} requre grad!'.format(name))

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        model.eval()
        for layer_id in self._config.layer_names:
            if hasattr(model, layer_id+'_adapter'): 
                getattr(model, layer_id+'_adapter').train()

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            logits, feature_outputs = model(inputs)
            loss = cross_entropy(logits, targets)
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
        cnn_pred_all, target_all = [], []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs = inputs.cuda()
            outputs, feature_outputs = model(inputs)

            cnn_preds = torch.max(outputs, dim=1)[1].detach().cpu()

            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
            else:
                cnn_correct += cnn_preds.eq(targets).sum()
                total += len(targets)

        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            return cnn_pred_all, target_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc