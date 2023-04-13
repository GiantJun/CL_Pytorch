import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy
from torch.utils.data import DataLoader

from backbone.cnn_prompt_net import Class_CNNPromptNet
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8


class CNN_Class_Prompt(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = self._config.mode.split('|')
    
    def prepare_task_data(self, data_manager):
        self._cur_task = data_manager.nb_tasks - 1
        self._cur_classes = data_manager.get_task_size(0)
        self._total_classes = self._known_classes + self._cur_classes
        train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='train')
        test_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(test_dataset)))

        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = Class_CNNPromptNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path,
                    gamma=self._config.gamma, layer_names=self._config.layer_names, mode=self._mode)
            self._logger.info('Created Class-specific prompt net!')

        if 'class_plus_one' in self._mode: # for class-specific prompt
            self._logger.info('Applying class plus 1 in classification head!')
            self._network.update_fc(self._total_classes)
        else:
            self._network.update_fc(self._total_classes)

        if self._config.freeze_fe:
            self._network.freeze_FE()
        self._network = self._network.cuda()

        self._logger.info('Initializing prompt in network...')
        with torch.no_grad():
            self._network.eval()
            self._network(torch.rand(self._config.batch_size, 3, self._config.img_size, self._config.img_size).cuda())

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        for layer_id in self._config.layer_names:
            if hasattr(self._network, layer_id+'_prompt'):
                prompt_module = getattr(self._network, layer_id+'_prompt')
                self._logger.info('{} params: {} , trainable params: {}'.format(layer_id+'_prompt', 
                    sum(p.numel() for p in prompt_module), 
                    sum(p.numel() for p in prompt_module if p.requires_grad)))
            if hasattr(self._network, layer_id+'_conv_1x1'):
                conv_module = getattr(self._network, layer_id+'_conv_1x1')
                self._logger.info('{} params: {} , trainable params: {}'.format(layer_id+'_conv_1x1', count_parameters(conv_module), 
                    count_parameters(conv_module, True)))

        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} requre grad!'.format(name))

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct_losses, wrong_losses = 0., 0.
        correct, total = 0, 0
        model.eval()
        for layer_id in self._config.layer_names:
            if hasattr(model, layer_id+'_conv_1x1'): 
                getattr(model, layer_id+'_conv_1x1').train()

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            logits, feature_outputs = model.train_forward(inputs, targets)
            correct_prompt_logits = logits[:logits.shape[0]//2]
            wrong_prompt_logits = logits[logits.shape[0]//2:]
            if 'class_plus_one' in self._mode:
                correct_target = target2onehot(targets, self._total_classes)
                prompt_correct_target = torch.cat((torch.ones(targets.shape[0], 1).cuda(), correct_target), dim=1)
                correct_loss = binary_cross_entropy(torch.sigmoid(correct_prompt_logits), prompt_correct_target)
                wrong_loss = binary_cross_entropy(torch.sigmoid(wrong_prompt_logits), torch.zeros(targets.shape[0], self._total_classes+1).cuda())
                preds = torch.max(correct_prompt_logits[:,1:], dim=1)[1]
            else:
                correct_loss = binary_cross_entropy(torch.sigmoid(correct_prompt_logits), target2onehot(targets, self._total_classes))
                if 'limit_0.5' in self._mode:
                    wrong_loss = binary_cross_entropy(torch.sigmoid(wrong_prompt_logits), target2onehot(targets, self._total_classes)*0.5)
                elif 'limit_0' in self._mode:
                    wrong_loss = binary_cross_entropy(torch.sigmoid(wrong_prompt_logits), torch.zeros(targets.shape[0], self._total_classes).cuda())
                else:
                    raise ValueError('Unknown mode {}'.format(self._mode))
                preds = torch.max(correct_prompt_logits, dim=1)[1]

            loss = correct_loss + wrong_loss
            correct_losses += correct_loss.item()
            wrong_losses += wrong_loss.item()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Correct_Loss', correct_losses/len(train_loader), 'Wrong_Loss', wrong_losses/len(train_loader)]
        return model, train_acc, train_loss
    
    def _epoch_test(self, model, test_loader, ret_pred_target=False):
        cnn_correct, prompt_correct, total, cnn_correct_w_correct_prompt = 0, 0, 0, 0
        cnn_pred_all, target_all = [], []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs = inputs.cuda()
            outputs, feature_outputs = model(inputs)

            cnn_preds, prompt_preds, cnn_preds_w_correct_prompt = [], [], []
            if 'class_plus_one' in self._mode:
                # self._logger.info('class plus one test!')
                for i in range(outputs.shape[0]//self._total_classes):
                    temp_logits = outputs[i*self._total_classes:(i+1)*self._total_classes]
                    prompt_id = torch.argmax(temp_logits[:,0]).item()
                    class_id = torch.argmax(temp_logits[prompt_id][1:]).item()
                    cnn_preds.append(class_id)
                    prompt_preds.append(prompt_id)
                    cnn_preds_w_correct_prompt.append(torch.argmax(temp_logits[targets[i]][1:]).item())
            else:
                # self._logger.info('guess prompt with image!')
                for i in range(outputs.shape[0]//self._total_classes):
                    temp_logits = outputs[i*self._total_classes:(i+1)*self._total_classes]
                    global_id = torch.argmax(temp_logits).item()
                    prompt_id, class_id = global_id//self._total_classes, global_id % self._total_classes
                    cnn_preds.append(class_id)
                    prompt_preds.append(prompt_id)
                    cnn_preds_w_correct_prompt.append(torch.argmax(temp_logits[targets[i]]).item())
            
            cnn_preds = torch.tensor(cnn_preds, dtype=int)
            cnn_preds_w_correct_prompt = torch.tensor(cnn_preds_w_correct_prompt, dtype=int)
            prompt_preds = torch.tensor(prompt_preds, dtype=int)
            prompt_correct += prompt_preds.eq(targets).sum()

            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
            else:
                cnn_correct += cnn_preds.eq(targets).sum()
                cnn_correct_w_correct_prompt += cnn_preds_w_correct_prompt.eq(targets).sum()
                total += len(targets)

        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            return cnn_pred_all, target_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            prompt_test_acc = np.around(tensor2numpy(prompt_correct)*100 / total, decimals=2)
            self._logger.info("Prompt predict acc={}".format(prompt_test_acc))
            test_acc_w_correct_prompt = np.around(tensor2numpy(cnn_correct_w_correct_prompt)*100 / total, decimals=2)
            self._logger.info("Test acc with correct prompt={}".format(test_acc_w_correct_prompt))
            return test_acc