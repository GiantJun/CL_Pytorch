import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.multi_steps.finetune_il import Finetune_IL
from backbone.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy, count_parameters
from quadprog import solve_qp

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

class GEM(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._previous_task_dataloader = []
        self._grad_dims = []
        self._G = None
        # self._margin = config.margin if config.margin is not None else 0.5

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

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

        if self._cur_task > 0:
            self._previous_task_dataloader = []
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
        self._network = self._network.cuda()
        if len(self._grad_dims) == 0:
            for param in self._network.parameters(): # fc 要参与,否则性能相差很大
                self._grad_dims.append(param.data.numel())
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
                        loss = F.cross_entropy(logits[:, :temp_end], task_targets)
                    elif self._incre_type == 'til':
                        loss = F.cross_entropy(logits[:, temp_begin:temp_end], task_targets-temp_begin)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    self._store_grad(model, t)
                    temp_begin = temp_end
                
                # for cur_task
                inputs, targets = task_inputs, task_targets
                if self._incre_type == 'cil':
                    preds = torch.max(logits[:, :task_end], dim=1)[1]
                elif self._incre_type == 'til':
                    preds = torch.max(logits[:, task_begin:task_end], dim=1)[1] + task_begin

                # check if gradient violates constraints
                dotp = torch.mm(self._G[:, task_id].unsqueeze(0), self._G[:, :task_id])
                if (dotp < 0).sum() != 0:
                    # apply gem to project grad
                    self.project_grad()
                    # copy gradients back
                    self._overwrite_grad(model, self._G[:, task_id])
            else:
                inputs, targets = inputs.cuda(), targets.cuda()

                logits, feature_outputs = model(inputs)
                if self._incre_type == 'cil':
                    loss = F.cross_entropy(logits[:, :task_end], targets)
                    preds = torch.max(logits[:, :task_end], dim=1)[1]
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
    
    def project_grad(self, margin=0.5, epsilon=1e-3):
        """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
        """
        memories_np = self._G[:, :self._cur_task].cpu().t().double().numpy()
        gradient_np = self._G[:, self._cur_task].unsqueeze(1).cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * epsilon
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        self._G[:, self._cur_task] = torch.Tensor(x).view(-1)