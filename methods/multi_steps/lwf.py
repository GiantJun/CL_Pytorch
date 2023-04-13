import numpy as np
import torch
from torch.nn.functional import cross_entropy
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy

# init_epoch=200
# init_lr=0.1
# init_milestones=[60,120,160]
# init_lr_decay=0.1
# init_weight_decay=0.0005


# epochs = 250
# lrate = 0.1
# milestones = [60,120, 180,220]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4
# num_workers=8
# T=2
# lamda=3

class LwF(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = config.T
        self._lamda = config.lamda
        if self._incre_type != 'til':
            raise ValueError('LWF is a task incremental method!')

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        losses_clf = 0.
        losses_kd = 0.
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            loss_clf = cross_entropy(logits, targets)
            losses_clf += loss_clf.item()
            if self._cur_task == 0:
                loss = loss_clf
            else:
                loss_kd = self._KD_loss(logits[:,:self._known_classes],self._old_network(inputs)[0],self._T)
                losses_kd += loss_kd.item()
                loss = loss_clf + self._lamda*loss_kd

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
        train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', losses_clf/len(train_loader), 'Loss_kd', losses_kd/len(train_loader)]
        return model, train_acc, train_loss

    # def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
    #     cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
    #     task_id_predict_correct = 0
    #     cnn_pred_all, target_all = [], []
    #     cnn_max_scores_all = []
    #     model.eval()
    #     for _, inputs, targets in test_loader:
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #         logits, feature_outputs = model(inputs)
    #         if self._incre_type == 'cil':
    #             known_class_num, total_class_num = 0, 0
    #             task_pred, task_max_score = [], []
    #             task_id_targets = torch.zeros(targets.shape[0], dtype=int).cuda()
    #             for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]):
    #                 total_class_num += cur_class_num

    #                 max_scores, pred = torch.max(torch.softmax(logits[:, known_class_num: total_class_num], dim=-1), dim=-1)
    #                 pred += known_class_num
    #                 task_pred.append(pred)
    #                 task_max_score.append(max_scores)

    #                 # generate task_id targets
    #                 task_data_idxs = torch.argwhere(torch.logical_and(targets>=known_class_num, targets<total_class_num)).squeeze(-1)
    #                 if len(task_data_idxs) > 0:
    #                     task_id_targets[task_data_idxs] = id

    #                 known_class_num = total_class_num
                
    #             task_pred = torch.stack(task_pred, dim=0) # [task_num, b]
    #             task_max_score = torch.stack(task_max_score, dim=0) # [task_num, b]
    #             task_id_pred = torch.max(task_max_score, dim=0)[1]
    #             cnn_preds = task_pred.gather(0, task_id_pred.unsqueeze(0)).squeeze(0)
    #             cnn_max_scores = task_max_score.gather(0, task_id_pred.unsqueeze(0)).squeeze(0)

    #             task_id_predict_correct += task_id_pred.eq(task_id_targets).cpu().sum()

    #         elif self._incre_type == 'til':
    #             cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits[:, task_begin:task_end], dim=-1), dim=-1)
    #             cnn_preds = cnn_preds + task_begin
                
    #         if ret_pred_target:
    #             cnn_pred_all.append(tensor2numpy(cnn_preds))
    #             target_all.append(tensor2numpy(targets))
    #             cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
    #         else:
    #             if ret_task_acc:
    #                 task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
    #                 cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]).cpu().sum()
    #                 task_total += len(task_data_idxs)
    #             cnn_correct += cnn_preds.eq(targets).cpu().sum()
            
    #         total += len(targets)
        
    #     if self._incre_type == 'cil':
    #         self._logger.info('Test task id predict acc (CNN) : {:.2f}'.format(np.around(task_id_predict_correct*100 / total, decimals=2)))
        
    #     if ret_pred_target:
    #         cnn_pred_all = np.concatenate(cnn_pred_all)
    #         cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
    #         target_all = np.concatenate(target_all)
    #         return cnn_pred_all, None, cnn_max_scores_all, None, target_all
    #     else:
    #         test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
    #         if ret_task_acc:
    #             test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
    #             return test_acc, test_task_acc
    #         else:
    #             return test_acc

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()
        
    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred/T, dim=1)
        soft = torch.softmax(soft/T, dim=1)
        return -1 * torch.mul(soft, pred).sum()/pred.shape[0]
