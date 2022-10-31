import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import logging
from torch import nn
import torch
from utils.toolkit import DummyDataset
import copy

EPSILON = 1e-8

class ReplayBank:

    def __init__(self, config):
        self._method = config.method
        self._batch_size = config.batch_size
        self._num_workers = config.num_workers

        self._memory_size = config.memory_size
        # 有两种样本存储形式, 但都固定存储空间。一种是固定每一类数据存储样本的数量(为True时)
        # 另一种在固定存储空间中，平均分配每一类允许存储的样本数量
        self._fixed_memory = config.fixed_memory
        if self._fixed_memory:
            self._memory_per_class = config.memory_per_class = self._memory_size // config.total_class_num
        self._sampling_method = config.sampling_method # 采样的方式

        self._transform = None
        self._use_path = None

        self._data_memory = [] # 第一维长度为类别数，第二维为每一类允许存放的样本数
        self._vector_memory = []
        self._valid_memory = []

        self._class_means = []
    
    def store_samplers(self, dataset:DummyDataset, model):
        class_range = np.unique(dataset.targets)
        if self._fixed_memory:
            per_class = self._memory_per_class
        else:
            per_class = self._memory_size // (len(self._data_memory) + len(class_range))
            if len(self._data_memory) > 0:
                self.reduce_memory(per_class)

        dataset = copy.deepcopy(dataset)
        # initial replayBank's transform
        if self._transform == None:
            self._transform = copy.deepcopy(dataset.transform)
            self._use_path = dataset.use_path

        class_means = []
        logging.info('Re-calculating class means for stored classes...')
        for class_idx, class_samples in enumerate(self._data_memory):
            idx_dataset = DummyDataset(class_samples, np.full(len(class_samples), class_idx), self._transform, self._use_path)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

            idx_vectors = self._extract_vectors(model, idx_loader)
            idx_vectors = F.normalize(idx_vectors, dim=1)# 对特征向量做归一化
            mean = torch.mean(idx_vectors, dim=0)
            mean = F.normalize(mean, dim=0)
            class_means.append(mean.unsqueeze(0))
            logging.info('calculated class mean of class {}'.format(class_idx))

        logging.info('Constructing exemplars for the sequence of {} new classes...'.format(len(class_range)))
        for class_idx in class_range:
            class_data_idx = np.where(np.logical_and(dataset.targets >=class_idx, dataset.targets < class_idx + 1))[0]
            idx_data, idx_targets = dataset.data[class_data_idx], dataset.targets[class_data_idx]
            idx_dataset = Subset(dataset, class_data_idx)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            idx_vectors = self._extract_vectors(model, idx_loader)
            if self._sampling_method == 'herding':
                selected_idx = self.herding_select(idx_vectors, per_class)
            elif self._sampling_method == 'random':
                selected_idx = self.random_select(idx_vectors, per_class)
            elif self._sampling_method == 'closest_to_mean':
                selected_idx = self.closest_to_mean_select(idx_vectors, per_class)
            logging.info("New Class {} instance will be stored: {} => {}".format(class_idx, len(idx_targets), per_class))
            
            # 计算类中心
            mean = torch.mean(idx_vectors[selected_idx], dim=0)
            mean = F.normalize(mean, dim=0)
            class_means.append(mean.unsqueeze(0))
            logging.info('calculated class mean of class {}'.format(class_idx))

            self._data_memory.append(idx_data[selected_idx])
            self._vector_memory.append(idx_vectors[selected_idx].cpu().numpy())
        
        logging.info('Replay Bank stored {} classes, {} samples for each class'.format(len(self._data_memory), per_class))
        self._class_means = torch.cat(class_means)

    ##################### Sampler Methods #####################
    def random_select(self, vectors, m):
        idxes = np.arange(vectors.shape[0])
        np.random.shuffle(idxes)
        return idxes[:m]
    
    def closest_to_mean_select(self, vectors, m):
        normalized_vector = F.normalize(vectors, dim=1) # 对特征向量做归一化
        class_mean = torch.mean(normalized_vector, dim=0).unsqueeze(0)
        # class_mean = F.normalize(class_mean, dim=0)
        distences = torch.cdist(normalized_vector, class_mean).squeeze()
        return torch.argsort(distences)[:m].cpu()

    def herding_select(self, vectors, m):
        selected_idx = []
        all_idxs = list(range(vectors.shape[0]))
        nomalized_vector = F.normalize(vectors, dim=1) # 对特征向量做归一化
        class_mean = torch.mean(nomalized_vector, dim=0)
        for k in range(1, m+1):
            sub_vectors = nomalized_vector[all_idxs]
            S = torch.sum(nomalized_vector[selected_idx], dim=0)
            mu_p = (sub_vectors + S) / k
            i = torch.argmin(torch.norm(class_mean-mu_p, p=2, dim=1))
            selected_idx.append(all_idxs.pop(i))
        return selected_idx
    ###########################################################

    def reduce_memory(self, m):
        for i in range(len(self._data_memory)):
            logging.info("Old class {} storage will be reduced: {} => {}".format(i, len(self._data_memory[i]), m))
            self._data_memory[i] = self._data_memory[i][:m]
            self._vector_memory[i] = self._vector_memory[i][:m]
            # logging.info("类别 {} 存储样本数为: {}".format(i, len(self._data_memory[i])))

    def KNN_classify(self, vectors=None, model=None, loader=None):
        if model != None and loader != None:
            vectors = self._extract_vectors(model, loader)
        
        vectors = F.normalize(vectors, dim=1)# 对特征向量做归一化

        dists = torch.cdist(vectors, self._class_means, p=2)
        nme_predicts = torch.argmin(dists, dim=1)
        return nme_predicts

    def get_memory(self, ret_vectors=False):
        target= []
        for class_idx, class_samples in enumerate(self._data_memory):
            target.append(np.full(len(class_samples), class_idx))

        logging.info('Replay stored samples info: stored_class={} , samples_per_class={} , total={}'.format(
                                                len(target),len(target[0]), len(target)*len(target[0])))
        
        if ret_vectors:
            return np.concatenate(self._data_memory), np.concatenate(target), np.concatenate(self._vector_memory)
        else:
            return np.concatenate(self._data_memory), np.concatenate(target)
    
    def get_unified_sample_dataset(self, new_task_dataset:DummyDataset, model):
        class_range = np.unique(new_task_dataset.targets)
        if self._fixed_memory:
            per_class = self._memory_per_class
        else:
            per_class = self._memory_size // len(self._data_memory)
        logging.info('Getting unified samples from old and new classes, {} samples for each class'.format(len(self._data_memory), per_class))
        train_transform = new_task_dataset.transform

        balanced_data = []
        balanced_targets = []

        # get old tasks' memory
        memory_data, memory_target = self.get_memory(False)
        balanced_data.append(memory_data)
        balanced_targets.append(memory_target)

        old_class_range = np.unique(memory_target)
        # balanced new task data and targets
        for class_idx in class_range:
            if class_idx in old_class_range:
                continue
            class_data_idx = np.where(np.logical_and(new_task_dataset.targets >=class_idx, new_task_dataset.targets < class_idx + 1))[0]
            idx_data, idx_targets = new_task_dataset.data[class_data_idx], new_task_dataset.targets[class_data_idx]
            idx_dataset = Subset(new_task_dataset, class_data_idx)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            idx_vectors = self._extract_vectors(model, idx_loader)
            if self._sampling_method == 'herding':
                selected_idx = self.herding_select(idx_vectors, per_class)
            elif self._sampling_method == 'random':
                selected_idx = self.random_select(idx_vectors, per_class)
            elif self._sampling_method == 'closest_to_mean':
                selected_idx = self.closest_to_mean_select(idx_vectors, per_class)
            logging.info("New Class {} instance will be down-sample: {} => {}".format(class_idx, len(idx_targets), per_class))

            balanced_data.append(idx_data[selected_idx])
            balanced_targets.append(np.full(len(selected_idx), class_idx))
        
        balanced_data, balanced_targets = np.concatenate(balanced_data), np.concatenate(balanced_targets)
        return DummyDataset(balanced_data, balanced_targets, train_transform, self._use_path)

    def _extract_vectors(self, model, loader, ret_data=False):
        model.eval()
        vectors = []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                if isinstance(model, nn.DataParallel):
                    _vectors = model.module.extract_features(_inputs.cuda())
                else:
                    _vectors = model.extract_features(_inputs.cuda())
                vectors.append(_vectors)
            return torch.cat(vectors)