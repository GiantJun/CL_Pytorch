import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy, count_parameters

from backbone.vit_prompts import L2P as Prompt
from backbone.vit_zoo import ViTZoo

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--prompt_pool', type=int, default=None, help='size of prompt pool')
    parser.add_argument('--prompt_length', type=int, default=None, help='length of prompt')
    parser.add_argument('--shallow_or_deep', type=bool, default=None, help='true for shallow, false for deep')
    return parser

class L2P(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._prompt_pool = config.prompt_pool
        self._prompt_length = config.prompt_length
        self._shallow_or_deep = config.shallow_or_deep

        if self._incre_type != 'cil':
            raise ValueError('DualPrompt is a class incremental method!')
    
    def prepare_model(self, checkpoint=None):
        if self._network == None:
            prompt_module = Prompt(768, self._config.nb_tasks, self._prompt_pool, self._prompt_length, self._shallow_or_deep)
            self._network = ViTZoo(self._logger, prompt_module=prompt_module)
        
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

        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} require grad!'.format(name))

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, prompt_losses = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, prompt_loss = model(inputs, train=True)
            
            loss = prompt_loss.sum()
            prompt_losses += loss.sum()

            # ce with heuristic
            logits[:,:task_begin] = -float('inf')
            ce_loss = F.cross_entropy(logits, targets)
            loss += ce_loss
            ce_losses += ce_loss.item()

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
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader), 'Loss_prompt', prompt_losses/len(train_loader)]
        return model, train_acc, train_loss

    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        cnn_pred_all, target_all = [], []
        cnn_max_scores_all = []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)
            cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits[:,:task_end], dim=-1), dim=-1)
                
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
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
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            target_all = np.concatenate(target_all)
            return cnn_pred_all, None, cnn_max_scores_all, None, target_all, None
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                return test_acc, test_task_acc
            else:
                return test_acc