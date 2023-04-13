import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch
from utils.toolkit import DummyDataset, pil_loader
import copy
from PIL import Image

EPSILON = 1e-8

def cat_with_broadcast(data_list:list):
    max_dim = 0
    max_len = 0
    for item in data_list:
        max_len += item.shape[0]
        if max_dim < item.shape[1]:
            max_dim = item.shape[1]
    
    result = np.zeros((max_len, max_dim))
    idx = 0
    for item in data_list:
        result[idx:idx+item.shape[0],:item.shape[1]] = item
        idx += item.shape[0]
    
    return result

class ReplayBank:

    def __init__(self, config, logger):
        self._logger = logger
        self._apply_nme = config.apply_nme
        self._batch_size = config.batch_size
        self._num_workers = config.num_workers
        self._total_class_num = config.total_class_num

        self._memory_size = config.memory_size
        # 有两种样本存储形式, 但都固定存储空间。一种是固定每一类数据存储样本的数量(为True时)
        # 另一种在固定存储空间中，平均分配每一类允许存储的样本数量
        self._fixed_memory = config.fixed_memory
        self._sampling_method = config.sampling_method # 采样的方式
        if self._fixed_memory:
            if config.memory_per_class is not None:
                self._memory_per_class = config.memory_per_class # 预期每类保存的样本数量
            elif self._memory_size is not None:
                self._memory_per_class = self._memory_size // config.total_class_num
            else:
                raise ValueError('Value error in setting memory per class!')

        self._data_memory = np.array([])
        self._targets_memory = np.array([])
        self._class_sampler_info = [] # 列表中保存了每个类实际保存的样本数

        self._class_means = []
        self._num_seen_examples = 0
    
    @property
    def sample_per_class(self):
        return self._memory_per_class

    def get_class_means(self):
        return self._class_means
    
    def set_class_means(self, class_means):
        self._class_means = class_means

    def store_samples(self, dataset:DummyDataset, model):
        """dataset 's transform should be in test mode!"""
        class_range = np.unique(dataset.targets)
        assert min(class_range)+1 > len(self._class_sampler_info), "Store_samples's dataset should not overlap with buffer"
        if self._fixed_memory:
            per_class = self._memory_per_class
        else:
            self._memory_per_class = per_class = self._memory_size // (len(self._class_sampler_info) + len(class_range))
            if len(self._class_sampler_info) > 0:
                self.reduce_memory(per_class)

        # to reduce calculation when applying replayBank (expecially for some methods do not apply nme)
        if self._apply_nme:
            class_means = []
            memory_dataset = DummyDataset(self._data_memory, self._targets_memory, dataset.transform, dataset.use_path)
            stored_data_means = self.cal_class_means(model, memory_dataset)
            if stored_data_means is not None:
                class_means.append(stored_data_means)

        data_mamory, targets_memory = [], []
        if len(self._class_sampler_info) > 0:
            data_mamory.append(self._data_memory)
            targets_memory.append(self._targets_memory)

        self._logger.info('Constructing exemplars for the sequence of {} new classes...'.format(len(class_range)))
        for class_idx in class_range:
            class_data_idx = np.where(dataset.targets == class_idx)[0]
            idx_data, idx_targets = dataset.data[class_data_idx], dataset.targets[class_data_idx]
            idx_dataset = Subset(dataset, class_data_idx)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            idx_vectors, idx_logits = self._extract_vectors(model, idx_loader)
            selected_idx = self.select_sample_indices(idx_vectors, per_class)
            self._logger.info("New Class {} instance will be stored: {} => {}".format(class_idx, len(idx_targets), len(selected_idx)))
            
            # to reduce calculation when applying replayBank (expecially for some methods do not apply nme)
            if self._apply_nme:
                # 计算类中心
                idx_vectors = F.normalize(idx_vectors[selected_idx], dim=1)# 对特征向量做归一化
                mean = torch.mean(idx_vectors, dim=0)
                mean = F.normalize(mean, dim=0)
                class_means.append(mean.unsqueeze(0))
                self._logger.info('calculated class mean of class {}'.format(class_idx))

            data_mamory.append(idx_data[selected_idx])
            targets_memory.append(idx_targets[selected_idx])
            
            self._class_sampler_info.append(len(selected_idx))
        
        self._logger.info('Replay Bank stored {} classes, {} samples ({} samples for each class)'.format(
                len(self._class_sampler_info), sum(self._class_sampler_info), per_class))
        
        if self._apply_nme:
            self._class_means = torch.cat(class_means, dim=0)
        
        self._data_memory = np.concatenate(data_mamory)
        self._targets_memory = np.concatenate(targets_memory)
    
    def cal_class_means(self, model, dataset:DummyDataset):
        class_means = []
        self._logger.info('Re-calculating class means for stored classes...')
        # for class_idx, class_samples in enumerate(self._data_memory):
        for class_idx in np.unique(dataset.targets):
            mask = np.where(dataset.targets == class_idx)[0]
            idx_dataset = DummyDataset(dataset.data[mask], dataset.targets[mask], dataset.transform, dataset.use_path)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

            idx_vectors, _ = self._extract_vectors(model, idx_loader)
            idx_vectors = F.normalize(idx_vectors, dim=1)# 对特征向量做归一化
            mean = torch.mean(idx_vectors, dim=0)
            mean = F.normalize(mean, dim=0)
            class_means.append(mean)
            self._logger.info('calculated class mean of class {}'.format(class_idx))
        return torch.stack(class_means, dim=0) if len(class_means) > 0 else None

    def reduce_memory(self, m):
        data_mamory, targets_memory = [], []
        for i in range(len(self._class_sampler_info)):
            if self._class_sampler_info[i] > m:
                store_sample_size = m
            else:
                self._logger.info('The whole class samples are less than the allocated memory size!')
                store_sample_size = self._class_sampler_info[i]
            
            self._logger.info("Old class {} storage will be reduced: {} => {}".format(i, self._class_sampler_info[i], store_sample_size))
            mask = np.where(self._targets_memory == i)[0]
            data_mamory.append(self._data_memory[mask[:store_sample_size]])
            targets_memory.append(self._targets_memory[mask[:store_sample_size]])
            self._class_sampler_info[i] = store_sample_size
            # self._logger.info("类别 {} 存储样本数为: {}".format(i, len(self._data_memory[i])))
        self._data_memory = np.concatenate(data_mamory)
        self._targets_memory = np.concatenate(targets_memory)

    def KNN_classify(self, vectors=None, model=None, loader=None, ret_logits=False):
        assert self._apply_nme, 'if apply_nme=False, you should not apply KNN_classify!'
        if model != None and loader != None:
            vectors, _ = self._extract_vectors(model, loader)
        
        vectors = F.normalize(vectors, dim=1)# 对特征向量做归一化

        dists = torch.cdist(vectors, self._class_means, p=2)
        min_scores, nme_predicts = torch.min(dists, dim=1)
        if ret_logits:
            return nme_predicts, dists
        else:
            return nme_predicts, 1-min_scores

    def get_memory(self, indices=None):
        replay_data, replay_targets = [], []
        if sum(self._class_sampler_info) <= 0:
            self._logger.info('Replay nothing or Nothing have been stored')
            return np.array([]), np.array([]), np.array([])
        elif indices is None: # default replay all stored data
            indices = range(len(self._class_sampler_info))

        for idx in indices:
            mask = np.where(self._targets_memory == idx)[0]
            replay_data.append(self._data_memory[mask])
            replay_targets.append(self._targets_memory[mask])
        
        return np.concatenate(replay_data), np.concatenate(replay_targets)
    
    def get_unified_sample_dataset(self, new_task_dataset:DummyDataset, model):
        """dataset 's transform should be in train mode!"""
        class_range = np.unique(new_task_dataset.targets)
        if self._fixed_memory:
            per_class = self._memory_per_class
        else:
            per_class = self._memory_size // len(self._class_sampler_info)
        self._logger.info('Getting unified samples from old and new classes, {} samples for each class (replay {} old classes)'.format(per_class, len(self._class_sampler_info)))

        balanced_data = []
        balanced_targets = []
        
        balanced_data.append(self._data_memory)
        balanced_targets.append(self._targets_memory)

        # balanced new task data and targets
        for class_idx in class_range:
            if class_idx < len(self._class_sampler_info):
                continue
            class_data_idx = np.where(np.logical_and(new_task_dataset.targets >=class_idx, new_task_dataset.targets < class_idx + 1))[0]
            idx_data, idx_targets = new_task_dataset.data[class_data_idx], new_task_dataset.targets[class_data_idx]
            idx_dataset = Subset(new_task_dataset, class_data_idx)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            idx_vectors, idx_logits = self._extract_vectors(model, idx_loader)
            selected_idx = self.select_sample_indices(idx_vectors, per_class)
            self._logger.info("New Class {} instance will be down-sample: {} => {}".format(class_idx, len(idx_targets), len(selected_idx)))

            balanced_data.append(idx_data[selected_idx])
            balanced_targets.append(idx_targets[selected_idx])
        
        balanced_data, balanced_targets = np.concatenate(balanced_data), np.concatenate(balanced_targets)
        return DummyDataset(balanced_data, balanced_targets, new_task_dataset.transform, new_task_dataset.use_path)
    
    def store_samples_reservoir(self, dataset:DummyDataset, model):
        """dataset 's transform should be in train mode!"""
        if not hasattr(self, '_soft_targets_memory'):
            self._soft_targets_memory = np.array([])
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        # logits = self._extract_vectors(model, loader)[1].detach().cpu().numpy()
        logits, data = self._extract_vectors(model, loader, ret_data=True)[1:]
        logits, data = logits.detach().cpu().numpy(), data.cpu().numpy()
        init_size = 0
        if len(self._data_memory) == 0:
            init_size = min(len(dataset), self._memory_size)
            self._data_memory = data[:init_size]
            self._targets_memory = dataset.targets[:init_size]
            self._soft_targets_memory = logits[:init_size]
            self._num_seen_examples += init_size
        elif len(self._data_memory) < self._memory_size:
            init_size = self._memory_size - len(self._data_memory)
            self._data_memory = np.concatenate([self._data_memory, data[:init_size]])
            self._targets_memory = np.concatenate([self._targets_memory, dataset.targets[:init_size]])
            self._soft_targets_memory = np.concatenate([self._soft_targets_memory, logits[:init_size]])
            self._num_seen_examples += init_size

        for i in range(init_size, len(dataset)):
            index = np.random.randint(0, self._num_seen_examples + 1)
            self._num_seen_examples += 1
            if index < self._memory_size:
                self._data_memory[index] = data[i]
                self._targets_memory[index] = dataset.targets[i]
                self._soft_targets_memory[index] = logits[i]

    def get_memory_reservoir(self):
        choice = np.random.choice(min(self._num_seen_examples, self._memory_size), size=self._batch_size, replace=False)

        data_all = torch.from_numpy(self._data_memory[choice])
        targets_all = torch.from_numpy(self._targets_memory[choice])
        soft_targets_all = torch.from_numpy(self._soft_targets_memory[choice])

        return (data_all, targets_all, soft_targets_all)

    ##################### Sampler Methods #####################
    def select_sample_indices(self, vectors, m):
        if self._sampling_method == 'herding':
            selected_idx = self.herding_select(vectors, m)
        elif self._sampling_method == 'random':
            selected_idx = self.random_select(vectors, m)
        elif self._sampling_method == 'closest_to_mean':
            selected_idx = self.closest_to_mean_select(vectors, m)
        else:
            raise ValueError('Unknown sample select strategy: {}'.format(self._sampling_method))
        return selected_idx

    def random_select(self, vectors, m):
        idxes = np.arange(vectors.shape[0])
        np.random.shuffle(idxes)# 防止类别数过少的情况
        
        # 防止类别数过少的情况
        if vectors.shape[0] > m:
            store_sample_size = m
        else:
            self._logger.info('The whole class samples are less than the allocated memory size!')
            store_sample_size = vectors.shape[0]
        return idxes[:store_sample_size]
    
    def closest_to_mean_select(self, vectors, m):
        normalized_vector = F.normalize(vectors, dim=1) # 对特征向量做归一化
        class_mean = torch.mean(normalized_vector, dim=0)
        class_mean = F.normalize(class_mean, dim=0).unsqueeze(0)
        distences = torch.cdist(normalized_vector, class_mean).squeeze()

        # 防止类别数过少的情况
        if vectors.shape[0] > m:
            store_sample_size = m
        else:
            self._logger.info('The whole class samples are less than the allocated memory size!')
            store_sample_size = vectors.shape[0]
        return torch.argsort(distences)[:store_sample_size].cpu()

    def herding_select(self, vectors, m):
        selected_idx = []
        all_idxs = list(range(vectors.shape[0]))
        nomalized_vector = F.normalize(vectors, dim=1) # 对特征向量做归一化
        class_mean = torch.mean(nomalized_vector, dim=0)
        # class_mean = F.normalize(class_mean, dim=0)

        # 防止类别数过少的情况
        if vectors.shape[0] > m:
            store_sample_size = m
        else:
            self._logger.info('The whole class samples are less than the allocated memory size!')
            store_sample_size = vectors.shape[0]
            
        for k in range(1, store_sample_size+1):
            sub_vectors = nomalized_vector[all_idxs]
            S = torch.sum(nomalized_vector[selected_idx], dim=0)
            mu_p = (sub_vectors + S) / k
            i = torch.argmin(torch.norm(class_mean-mu_p, p=2, dim=1))
            selected_idx.append(all_idxs.pop(i))
        return selected_idx
    ###########################################################

    def _extract_vectors(self, model, loader, ret_data=False):
        model.eval()
        vectors = []
        logits = []
        data = []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                out, output_features = model(_inputs.cuda())
                vectors.append(output_features['features'])
                logits.append(out)
                if ret_data:
                    data.append(_inputs)
            if ret_data:
                return torch.cat(vectors), torch.cat(logits), torch.cat(data)
            else:
                return torch.cat(vectors), torch.cat(logits)
    