import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from methods.multi_steps.finetune_il import Finetune_IL
from backbone.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy, count_parameters
from sklearn.decomposition import PCA

# import random
from sklearn.model_selection import StratifiedShuffleSplit

# init_epoch=200
# init_lr=0.1
# init_milestones=[60,120,170]
# init_lr_decay=0.1
# init_weight_decay=0.0005

# epochs=100
# lrate = 0.1
# milestones=[30,60,80]
# lrate_decay = 0.1
# batch_size=128
# weight_decay=2e-4
# num_workers=4

class LOGD_GEM(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._previous_task_dataloader = []
        self._grad_dims = []
        self._G = None
        self._block_grad_dims = [] # different from gem
        # self._margin = config.margin if config.margin is not None else 0.5
        
        # avoid logging lots of 'RuntimeWarning: invalid value encountered in true_divide' in pca.fit()
        np.seterr(divide='ignore',invalid='ignore')

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='train') # mode=test?
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')

        if self._cur_task > 0:
            self._previous_task_dataloader = []
            if self._incre_type == 'cil' and self._cur_task > 1:
                previous_task_dataset = data_manager.get_dataset(indices=[], appendent=self._memory_bank.get_memory(np.arange(0, self._known_classes)),
                        source='train', mode='train')
                sample_idxs = list(range(len(previous_task_dataset.targets)))

                # episodic memory 中每一类保证均衡
                kfold = StratifiedShuffleSplit(n_splits=self._cur_task, random_state=0)
                for train_idxs, test_idxs in kfold.split(sample_idxs, previous_task_dataset.targets):
                    sampler = SubsetRandomSampler(indices=test_idxs)
                    previous_dataloader = DataLoader(previous_task_dataset, batch_size=len(sampler), 
                        num_workers=self._num_workers, sampler=sampler)
                    self._previous_task_dataloader.append(previous_dataloader)

                # random.shuffle(sample_idxs)
                # sample_per_class = self._memory_bank.sample_per_class
                # # 此处假设每类保存的样本数一样，暂时不考虑每类实际存储样本数与预期不一样的情况
                # for previous_task_id in range(self._cur_task):
                #     begin, end = sum(self._increment_steps[:previous_task_id]), sum(self._increment_steps[:previous_task_id+1])
                #     sampler = SubsetRandomSampler(sample_idxs[begin*sample_per_class:end*sample_per_class])
                #     previous_dataloader = DataLoader(previous_task_dataset, batch_size=len(sampler), 
                #         num_workers=self._num_workers, sampler=sampler)
                #     self._previous_task_dataloader.append(previous_dataloader)
            else:
                for previous_task_id in range(self._cur_task):
                    begin, end = sum(self._increment_steps[:previous_task_id]), sum(self._increment_steps[:previous_task_id+1])
                    previous_task_dataset = data_manager.get_dataset(indices=[], appendent=self._memory_bank.get_memory(np.arange(begin, end)),
                        source='train', mode='train')
                    previous_dataloader = DataLoader(previous_task_dataset, batch_size=len(previous_task_dataset), num_workers=self._num_workers)
                    self._previous_task_dataloader.append(previous_dataloader)

    
    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path).cuda()
            self._network.update_fc(sum(self._increment_steps))

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        if self._config.freeze_fe:
            self._network.freeze_FE()

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        if self._gpu_num > 1:
            self._network = nn.DataParallel(self._network, list(range(self._gpu_num)))
        self._network = self._network.cuda()
        if len(self._grad_dims) == 0:
            for index, param in enumerate(self._network.parameters()): # fc 要参与,否则性能相差很大
                self._grad_dims.append(param.data.numel())
                if index % 5 == 0:
                    block_params = 0
                    block_params += param.data.numel()
                    if index == len(list(self._network.parameters())) - 1:
                        self._block_grad_dims.append(block_params)
                elif index % 5 == 4:
                    block_params += param.data.numel()
                    self._block_grad_dims.append(block_params)
                else:
                    block_params += param.data.numel()
                    if index == len(list(self._network.parameters())) - 1:
                        self._block_grad_dims.append(block_params)

            self._G = torch.zeros((sum(self._grad_dims), self._nb_tasks)).cuda()
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            
            if task_id > 0:
                temp_begin, temp_end = 0, 0
                # store gradients
                for t in range(task_id+1):
                    if t == task_id:
                        task_inputs, task_targets = inputs, targets
                    else:
                        _, task_inputs, task_targets = next(self._previous_task_dataloader[t].__iter__())

                    task_inputs, task_targets = task_inputs.cuda(), task_targets.cuda()
                    logits, feature_outputs = model(task_inputs)
                    
                    temp_end = temp_begin + self._increment_steps[t]
                    if self._incre_type == 'cil':
                        loss = F.cross_entropy(logits[:, :task_end], task_targets)
                    elif self._incre_type == 'til':
                        loss = F.cross_entropy(logits[:, temp_begin:temp_end], task_targets-temp_begin)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    self._store_grad(model, t)
                    temp_begin = temp_end
                
                # for cur_task
                inputs, targets = task_inputs, task_targets
                if self._incre_type == 'cil':
                    preds = torch.max(logits[:,:task_end], dim=1)[1]
                elif self._incre_type == 'til':
                    preds = torch.max(logits[:, task_begin:task_end], dim=1)[1] + task_begin

                # check if gradient violates constraints
                for count in range(len(self._block_grad_dims)):
                    begin = 0 if count == 0 else sum(self._block_grad_dims[:count])
                    end = sum(self._block_grad_dims[:count+1])
                    if begin == end:
                        continue
                    self.project_grad(begin, end, task_id=task_id)
                
                # copy gradients back
                self._overwrite_grad(model, self._G[:, task_id])
            else:
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

    def _store_grad(self, model, task_id):
        self._G[:, task_id].fill_(0.0)
        cnt = 0
        for param in model.parameters():
            if param.grad is not None:
                begin = 0 if cnt == 0 else sum(self._grad_dims[:cnt])
                end = sum(self._grad_dims[:cnt + 1])
                self._G[begin: end, task_id].copy_(param.grad.data.view(-1))
            cnt += 1

    def _overwrite_grad(self, model, newgrad):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in model.parameters():
            if param.grad is not None:
                begin = 0 if cnt == 0 else sum(self._grad_dims[:cnt])
                end = sum(self._grad_dims[:cnt + 1])
                this_grad = newgrad[begin: end].contiguous().view(param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1
    
    def project_grad(self, begin, end, task_id, margin=0.5, epsilon=1e-3):
        """
        First we do orthognality of memories and then find the null space of
        these memories gradients
        """
        # ########### CPU version(origin implementation) begin ###############
        # memories_np = self._G[begin:end, :task_id].cpu().t().double().numpy() # [task_num-1, features]
        # gradient_np = self._G[begin:end, task_id].cpu().contiguous().view(-1).double().numpy() # [features]
        # memories_np_mean = np.mean(memories_np, axis=0) # 获得 shared gradient, [features]
        # memories_np_sum = np.sum(memories_np, axis=0) # [features]
        # memories_np_del_mean = memories_np - memories_np_mean.reshape(1, -1) # 获得 task-specific gradient, [task_num-1, features]
        
        # memories_np_pca = PCA(n_components=min(3, len(memories_np)))
        # memories_np_pca.fit(memories_np_del_mean)
        # if len(memories_np) == 1:
        #     x = gradient_np - np.min([
        #         (memories_np_sum.transpose().dot(gradient_np) /
        #         memories_np_sum.transpose().dot(memories_np_sum)), -margin
        #     ]) * memories_np_sum
        # else:
        #     # memories_np_pca = orth(memories_np_del_mean.transpose()).transpose()

        #     memories_np_orth = memories_np_pca.components_ # [n_components, features]
        #     # memories_np_orth = memories_np_pca
        #     Pg = gradient_np - memories_np_orth.transpose().dot(
        #         memories_np_orth.dot(gradient_np))
        #     Pg_bar = memories_np_sum - memories_np_orth.transpose().dot(
        #         memories_np_orth.dot(memories_np_sum))
        #     if memories_np_sum.transpose().dot(Pg) > 0: # 求 w 中的判断条件
        #         x = Pg
        #     else:
        #         x = gradient_np - np.min([
        #             memories_np_sum.transpose().dot(Pg) /
        #             memories_np_sum.transpose().dot(Pg_bar), -margin
        #         ]) * memories_np_sum - memories_np_orth.transpose().dot(
        #             memories_np_orth.dot(gradient_np)) + memories_np_sum.transpose(
        #             ).dot(Pg) / memories_np_sum.transpose().dot(
        #                 Pg_bar) * memories_np_orth.transpose().dot(
        #                     memories_np_orth.dot(memories_np_sum))
        
        # self._G[begin:end, task_id] = torch.Tensor(x).view(-1)
        # ########### CPU version(origin implementation) end ###############


        ########### GPU version(origin implementation) begin ###############
        memories_np = self._G[begin:end, :task_id].t().double() # [task_num-1, features]
        gradient_np = self._G[begin:end, task_id].contiguous().view(-1).double() # [features]
        memories_np_mean = torch.mean(memories_np, dim=0) # 获得 shared gradient, [features]
        memories_np_sum = torch.sum(memories_np, dim=0) # [features]
        memories_np_del_mean = memories_np - memories_np_mean.reshape(1, -1) # 获得 task-specific gradient, # [task_num-1, features]
        
        if len(memories_np) == 1:
            x = gradient_np - min(
                (memories_np_sum.dot(gradient_np) /
                memories_np_sum.dot(memories_np_sum)), -margin
            ) * memories_np_sum
        else:
            # memories_np_pca = orth(memories_np_del_mean.t()).t()

            memories_np_orth = torch.pca_lowrank(memories_np_del_mean, q=min(3, len(memories_np)))[2].t() # [n_components, features]
            # memories_np_orth = memories_np_pca
            Pg = gradient_np - torch.mm(memories_np_orth.t(), torch.mm(memories_np_orth, gradient_np.unsqueeze(1))).squeeze(1) # [features]
            Pg_bar = memories_np_sum - torch.mm(memories_np_orth.t(), torch.mm(memories_np_orth, memories_np_sum.unsqueeze(1))).squeeze(1) # [features]
            if memories_np_sum.dot(Pg) > 0: # 求 w 中的判断条件
                x = Pg
            else:
                x = gradient_np - min(
                    memories_np_sum.dot(Pg) /
                    memories_np_sum.dot(Pg_bar), -margin
                ) * memories_np_sum - torch.mm(memories_np_orth.t(),
                    torch.mm(memories_np_orth, gradient_np.unsqueeze(1))).squeeze(1) + memories_np_sum.dot(Pg) / memories_np_sum.dot(
                        Pg_bar) * torch.mm(memories_np_orth.t(), torch.mm(memories_np_orth, memories_np_sum.unsqueeze(1))).squeeze(1)
        
        self._G[begin:end, task_id] = x.view(-1)
        ########### GPU version(origin implementation) end ###############