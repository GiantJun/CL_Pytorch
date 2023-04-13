import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from backbone.cnn_prompt_net import ProtoNetV2
from methods.single_step.finetune_normal import Finetune_normal
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8


class CNN_Prompt_v3(Finetune_normal):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = self._config.mode.split('|')
        self._nb_proxy = config.nb_proxy
        self.prompt = []
 
    def prepare_task_data(self, data_manager):
        self._cur_task = self._nb_tasks-1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        
        train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        test_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        
        self._logger.info('Train dataset size: {}'.format(len(train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(test_dataset)))

        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        
        # initial prompt
        trsf = transforms.Compose([*data_manager._common_trsf])
        if len(self.prompt) == 0:
            for class_id in range(self._total_classes):
                imgs = []
                img_data = data_manager.get_class_sample(class_id, self._nb_proxy)
                for img_id in range(len(img_data)):
                    if 'rand_init' in self._mode:
                        imgs.append(torch.rand_like(trsf(img_data[img_id]).unsqueeze(0)))
                    else:
                        imgs.append(trsf(img_data[img_id]).unsqueeze(0))
                self.prompt.append(torch.cat(imgs, dim=0))
            self._logger.info('Inited img prompt')


    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = ProtoNetV2(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path, self._mode, self._config.use_MLP)
        # init prompt in the network
        self._network.init_prompt(self.prompt)
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        if self._config.freeze_fe:
            self._network.freeze_FE()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} require grad !'.format(name))
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        
        model.eval()
        i = 0
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            if 'min_enhance' in self._mode:
                print(i)
                i += 1
                if i == 39:
                    print('stop here!')
                logits, feature_outputs = model(inputs, targets)
            else:
                logits, feature_outputs = model(inputs)

            if 'bce' in self._mode:
                loss = F.binary_cross_entropy(logits, target2onehot(targets))
            else:    
                loss = F.cross_entropy(logits, targets)
            
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
        if self._config.freeze_fe:
            model.eval()
        else:
            model.train()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs, feature_outputs = model(inputs)

            cnn_preds = torch.max(outputs, dim=1)[1]
            
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