import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from backbone.inc_net import IncrementalNet
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy, bn_track_stats, pil_loader
from PIL import Image
from utils.losses import SupConLoss

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--alpha', type=float, default=None, help='balance coeeficient in loss terms')
    parser.add_argument('--beta', type=float, default=None, help='balance coeeficient in loss terms')
    parser.add_argument('--lamda', type=float, default=None, help='balance coeeficient in loss terms')
    parser.add_argument('--gamma', type=float, default=None, help='balance coeeficient in loss terms')
    parser.add_argument('--eta', type=float, default=None, help='balance coeeficient in loss terms')
    
    parser.add_argument('--margin', type=float, default=None, help='margin for xder')
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits for simclr')
    return parser

class X_DER(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._alpha = config.alpha
        self._beta = config.beta
        self._gamma = config.gamma
        self._lamda = config.lamda
        self._margin = config.margin
        self._eta = config.eta
        
        self._T = config.T
        self.strong_aug = None
        self._supConLoss = SupConLoss(temperature=self._T, base_temperature=self._T, reduction='sum')

        self._update_counter = np.zeros(self._memory_size)

        if self._incre_type != 'cil' and self._incre_type != 'til':
            raise ValueError('X-DER is a class/task incremental method!')
    
    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
            self._network.update_fc(self._config.total_class_num)

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            # if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
            #     self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        if self.strong_aug is None:
            self.strong_aug = data_manager.get_strong_transform()

        self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='train')
        self._train_dataset.set_ret_origin(True)
        
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

    def store_samples(self):
        self._memory_bank.store_samples_reservoir_v2(self._train_dataset, self._network, self._gamma)
        self._update_counter = np.zeros(self._memory_size)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, der_losses, derpp_losses = 0., 0., 0.
        supCon_losses, constr_past_losses, constr_futu_losses = 0., 0., 0.
        correct, total = 0, 0
        model.train()
        for no_aug_inputs, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            # Present head
            ce_loss = F.cross_entropy(logits[:, task_begin:task_end], targets-task_begin)
            loss = ce_loss
            ce_losses += ce_loss.item()
            
            if not self._memory_bank.is_empty():
                # Distillation Replay Loss (all heads)
                buf_idx1, buf_inputs1, buf_targets1, buf_soft_targets1 = self._memory_bank.get_memory_reservoir(self._replay_batch_size,
                                                                                                                self._train_dataset.use_path,
                                                                                                                self._train_dataset.transform,
                                                                                                                ret_idx=True)
                buf_inputs1, buf_targets1 = buf_inputs1.cuda(), buf_targets1.cuda()
                buf_soft_targets1 = buf_soft_targets1.cuda()

                buf_outputs1 = model(buf_inputs1)[0]
                der_loss = F.mse_loss(buf_outputs1, buf_soft_targets1, reduction='none').mean() * self._alpha
                loss += der_loss
                der_losses += der_loss.item()

                # Label Replay Loss (past heads)
                buf_idx2, buf_inputs2, buf_targets2, buf_soft_targets2 = self._memory_bank.get_memory_reservoir(self._replay_batch_size,
                                                                                                                self._train_dataset.use_path,
                                                                                                                self._train_dataset.transform,
                                                                                                                ret_idx=True)
                buf_inputs2, buf_targets2 = buf_inputs2.cuda(), buf_targets2.cuda()
                buf_soft_targets2 = buf_soft_targets2.cuda()

                buf_outputs2 = model(buf_inputs2)[0]
                derpp_loss = F.cross_entropy(buf_outputs2[:, :task_begin], buf_targets2) * self._beta
                loss += derpp_loss
                derpp_losses += derpp_loss.item()

                ###### update future past logits (begin) ######
                # Merge Batches & Remove Duplicates
                buf_idx = torch.cat([buf_idx1, buf_idx2]) # cpu
                buf_targets = torch.cat([buf_targets1, buf_targets2]) # gpu
                buf_outputs = torch.cat([buf_outputs1, buf_outputs2]) # gpu
                eyey = torch.eye(self._memory_size)[buf_idx] # cpu
                # umask is a mask that mark every repeat sample to False, others to True
                umask = (eyey * eyey.cumsum(0)).sum(1) < 2 # umask.size = (len(buf_idx),)
                umask_gpu = umask.cuda()

                buf_idx = buf_idx[umask].numpy()
                buf_targets = buf_targets[umask_gpu]
                buf_outputs = buf_outputs[umask_gpu]

                chosen = buf_targets.cpu().numpy() < task_begin # choose old tasks logits (not include cur_task)
                self._update_counter[buf_idx[chosen]] += 1 # record update frequency
                c = chosen.copy()  # use c as a mask to only update old tasks
                # here is sampling. The sampled ones were less likely to be sampled again if they have been sampled
                chosen[c] = np.random.rand(*chosen[c].shape) * self._update_counter[buf_idx[c]] < 1
                
                if chosen.any():
                    self._memory_bank.update_memory_reservoir(buf_outputs.detach().cpu().numpy()[chosen], buf_idx[chosen],
                                                              task_begin, self._gamma)
                ###### update future past logits (end) ######
            
            # Future preparation with SupContrastive Loss (future heads)
            if task_id < self._nb_tasks - 1:
                scl_labels = targets[:self._replay_batch_size]
                scl_na_inputs = no_aug_inputs[:self._replay_batch_size]
                if not self._memory_bank.is_empty():
                    buf_na_inputsscl, buf_labelsscl, _ = self._memory_bank.get_memory_reservoir(self._replay_batch_size, 
                                                                                                self._train_dataset.use_path,
                                                                                                transform=None)
                    scl_na_inputs = torch.cat([buf_na_inputsscl, scl_na_inputs])
                    scl_labels = torch.cat([buf_labelsscl.cuda(), scl_labels])
                
                scl_inputs = []
                for img_info in scl_na_inputs.repeat_interleave(2, 0):
                    if self._train_dataset.use_path:
                        scl_inputs.append(self.strong_aug(pil_loader(img_info)))
                    else:
                        scl_inputs.append(self.strong_aug(Image.fromarray(img_info.numpy())))
                scl_inputs = torch.stack(scl_inputs).cuda()

                with bn_track_stats(model, False):
                    scl_outputs = model(scl_inputs)[0]

                scl_featuresFull = scl_outputs.reshape(-1, 2, scl_outputs.shape[-1])  # [N, n_aug, 100]

                supCon_loss = []
                future_task_begin = task_end
                for increment in self._increment_steps[task_id+1:]:
                    task_logits = scl_featuresFull[:, :, future_task_begin:future_task_begin+increment]
                    supCon_loss.append(self._supConLoss(features=F.normalize(task_logits, dim=2), labels=scl_labels) / increment)
                    future_task_begin += increment
                
                supCon_loss = torch.stack(supCon_loss).mean() * self._lamda
                loss += supCon_loss
                supCon_losses += supCon_loss.item()

            # Past Logits Constraint
            if task_id > 0:
                cur_head = torch.softmax(logits[:, :task_end], 1)

                good_head = cur_head[:, task_begin:task_end]
                bad_head = cur_head[:, :task_begin]

                constr_past_loss = bad_head.max(1)[0].detach() + self._margin - good_head.max(1)[0]

                mask = constr_past_loss > 0

                if (mask).any():
                    constr_past_loss = self._eta * constr_past_loss[mask].mean()
                    loss += constr_past_loss
                    constr_past_losses += constr_past_loss.item()

            # Future Logits Constraint
            if task_id < self._nb_tasks - 1:
                bad_head = logits[:, task_end:]
                good_head = logits[:, task_begin:task_end]

                constr_futu_loss = bad_head.max(1)[0] + self._margin - good_head.max(1)[0]

                if not self._memory_bank.is_empty():
                    bad_head = buf_outputs[:, task_end:]
                    bad_head_max = bad_head.max(1)[0]

                    good_head_y = torch.ones_like(bad_head_max, dtype=torch.int64) * -1
                    task_begin_temp = 0
                    for increment in self._increment_steps[:task_id+1]:
                        task_mask = torch.where(torch.logical_and((buf_targets >= task_begin_temp), (buf_targets < (task_begin_temp + increment))))[0]
                        if len(task_mask) > 0:
                            good_head_y[task_mask] = buf_outputs[task_mask][:, task_begin_temp:task_begin_temp+increment].max(1)[1] + task_begin_temp
                        task_begin_temp += increment
                    
                    constr_futu_loss_buf = bad_head_max + self._margin - buf_outputs[torch.arange(len(good_head_y)), good_head_y]
                    constr_futu_loss = torch.cat([constr_futu_loss, constr_futu_loss_buf])

                mask = constr_futu_loss > 0
                if (mask).any():
                    constr_futu_loss = self._eta * constr_futu_loss[mask].mean()
                    loss += constr_futu_loss
                    constr_futu_losses += constr_futu_loss.item()

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
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader), 'Loss_der', der_losses/len(train_loader),
                      'Loss_derpp', derpp_losses/len(train_loader), 'Loss_sup_contrast', supCon_losses/len(train_loader),
                      'Loss_constr_past', constr_past_losses/len(train_loader), 'Loss_constr_futu', constr_futu_losses/len(train_loader)]
        return model, train_acc, train_loss
