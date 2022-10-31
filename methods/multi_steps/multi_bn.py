import numpy as np
import torch
# import medmnist
# from medmnist import INFO
from torch import nn
from torch.utils.data import DataLoader
from methods.multi_steps.finetune_il import Finetune_IL
from backbone.inc_net import IncrementalNet
from utils.toolkit import accuracy, tensor2numpy
from utils.toolkit import count_parameters


EPSILON = 1e-8

class Multi_BN(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._network_list = []
        self._bn_type = config.bn_type

        # 虽然方法是 til 但为了方便使用 cil 做推理（主要在测试时用到这个参数）
        self._incre_type = 'cil'

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
    
        if self._cur_task == 0:
            self._network_list.append(self._network)
            self._network.update_fc(self._cur_classes)
        else:
            self._network = IncrementalNet(self._backbone, False)
            self._network_list.append(self._network)
            self._network.update_fc(self._cur_classes)
            state_dict = self._network.convnet.state_dict()

            #["default", "last", "first", "pretrained"]
            if self._bn_type == "default":
                self._logger.info("update_bn_with_default_setting")
                state_dict.update(self._network_list[self._cur_task - 1].convnet.state_dict())
                self._network.convnet.load_state_dict(state_dict)
                self.reset_bn(self._network.convnet)
            elif self._bn_type == "last":
                self._logger.info("update_bn_with_last_model")
                state_dict.update(self._network_list[self._cur_task - 1].convnet.state_dict())
                self._network.convnet.load_state_dict(state_dict)
            elif self._bn_type == "first":
                self._logger.info("update_bn_with_first_model")
                state_dict.update(self._network_list[0].convnet.state_dict())
                self._network.convnet.load_state_dict(state_dict)
            else:
                #to be finished
                self._logger.info("update_bn_with_pretrained_model")
                state_dict.update(self._network_list[self._cur_task - 1].convnet.state_dict())
                dst_dict = torch.load("./saved_parameters/imagenet200_simsiam_pretrained_model_bn.pth")
                state_dict.update(dst_dict)
                self._network.convnet.load_state_dict(state_dict)

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._logger.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        self._network.cuda()
        for name, param in self._network.named_parameters():
            if 'fc' in name or 'bn' in name:
                self._logger.info('{} require grad.'.format(name))
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = self._get_optimizer(self._network, self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        self._network = self._update_representation(self._network, train_loader, test_loader, optimizer, scheduler)
    
    def reset_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.reset_parameters()

    def _eval_cnn_nme_til(self, model, data_manager):
        known_classes = 0
        total_classes = 0
        cnn_pred_result, nme_pred_result, y_true_result = [], [], []
        for task_id in range(self._cur_task + 1):
            cur_classes = data_manager.get_task_size(task_id)
            total_classes += cur_classes
            test_dataset = data_manager.get_dataset(indices=np.arange(known_classes, total_classes), source='test', 
                                                    mode='test')
            test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            cnn_pred, nme_pred, y_true = self._eval_cnn_nme_cil(self._network_list[task_id], test_loader)
            cnn_pred_result.append(cnn_pred+known_classes)
            if len(nme_pred) > 0:
                nme_pred_result.append(nme_pred+known_classes)
            y_true_result.append(y_true)
            known_classes = total_classes

        if len(nme_pred_result) == 1:
            cnn_pred_result = cnn_pred_result[0]
            nme_pred_result = nme_pred_result[0]
            y_true_result = y_true_result[0]
        elif len(nme_pred_result) > 1:
            cnn_pred_result = np.concatenate(cnn_pred_result)
            nme_pred_result = np.concatenate(nme_pred_result)
            y_true_result = np.concatenate(y_true_result)
        else:
            cnn_pred_result = np.concatenate(cnn_pred_result)
            y_true_result = np.concatenate(y_true_result)

        return cnn_pred_result, nme_pred_result, y_true_result
    
    def _update_representation(self, model, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        for epoch in range(epochs):
            model.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()

                logits = model(inputs)['logits']
                loss = self._criterion(logits, targets - self._known_classes)
                preds = torch.max(logits, dim=1)[1] + self._known_classes
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
            self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc)
            
            self._tblog.add_scalar('seed{}_train/loss'.format(self._seed), losses/len(train_loader), self._history_epochs+epoch)
            self._tblog.add_scalar('seed{}_train/acc'.format(self._seed), train_acc, self._history_epochs+epoch)

            self._logger.info(info)
        
        self._history_epochs += epochs
        return model
    
    def eval_task(self, data_manager):
        if self.cnn_task_metric_curve == None:
            self.cnn_task_metric_curve = [[] for i in range(data_manager.nb_tasks)]
            self.nme_task_metric_curve = [[] for i in range(data_manager.nb_tasks)]

        self._logger.info(50*"-")
        self._logger.info("log {} of every task".format(self._eval_metric))
        self._logger.info(50*"-")
        cnn_pred, nme_pred, y_true = self._eval_cnn_nme_til(self._network, data_manager)

        # 计算top1, 这里去掉了计算 topk 的代码
        if self._eval_metric == 'acc':
            cnn_total, cnn_task = accuracy(cnn_pred.T, y_true, self._total_classes, data_manager._increments)
        else:
            pass
        self.cnn_metric_curve.append(cnn_total)
        self._logger.info("CNN : {} curve of all task is {}".format(self._eval_metric, self.cnn_metric_curve))
        for i in range(len(cnn_task)):
            self.cnn_task_metric_curve[i].append(cnn_task[i])
            self._logger.info("CNN : task {} {} curve is {}".format(i, self._eval_metric, self.cnn_task_metric_curve[i]))
        self._logger.info("CNN : Average Acc: {:.2f}".format(np.mean(self.cnn_metric_curve)))
        self._logger.info(' ')
    
        if len(nme_pred) != 0:
            if self._eval_metric == 'acc':
                nme_total, nme_task = accuracy(nme_pred.T, y_true, self._total_classes, data_manager._increments)
            else:
                pass
            self.nme_metric_curve.append(nme_total)
            self._logger.info("NME : {} curve of all task is {}".format(self._eval_metric, self.nme_metric_curve))
            for i in range(len(nme_task)):
                self.nme_task_metric_curve[i].append(nme_task[i])
                self._logger.info("NME : task {} {} curve is {}".format(i, self._eval_metric, self.nme_task_metric_curve[i]))
            self._logger.info("NME : Average Acc: {:.2f}".format(np.mean(self.nme_metric_curve)))
            self._logger.info(' ')