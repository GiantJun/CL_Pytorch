import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch
from utils.toolkit import DummyDataset

from utils.replayBank import ReplayBank

EPSILON = 1e-8

class ReplayBank_Split(ReplayBank):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self._increment_steps = config.increment_steps
        self._feature_dim = None
        self._split_class_means = []

    def cal_class_means_split(self, known_task_id, model, transform, use_path):
        if self._feature_dim is None:
            self._feature_dim = model.feature_dim

        buf_dataset = DummyDataset(self._data_memory, self._targets_memory, transform, use_path)
        buf_loader = DataLoader(buf_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        buf_features = self._extract_vectors(model, buf_loader)[0]

        split_class_means = []
        task_begin, task_end = 0, 0
        for task_id in range(known_task_id+1):
            task_head_class_mean = []
            task_end += self._increment_steps[task_id]
            task_head_features = buf_features[:, task_id*self._feature_dim: (task_id+1)*self._feature_dim]
            for class_id in range(task_begin, task_end):
                mask = np.where(self._targets_memory==class_id)[0]
                idx_vectors = F.normalize(task_head_features[mask], dim=1)# 对特征向量做归一化
                mean = torch.mean(idx_vectors, dim=0)
                mean = F.normalize(mean, dim=0)
                task_head_class_mean.append(mean)
            
            if known_task_id > 0: # others
                mask = np.where(np.logical_or(self._targets_memory<task_begin, self._targets_memory>=task_end))
                idx_vectors = F.normalize(task_head_features[mask], dim=1)# 对特征向量做归一化
                mean = torch.mean(idx_vectors, dim=0)
                mean = F.normalize(mean, dim=0)
                task_head_class_mean.append(mean)

            split_class_means.append(torch.stack(task_head_class_mean))

            task_begin = task_end
            self._logger.info('calculated class mean of part {}'.format(task_id))
        
        self._split_class_means = split_class_means

    def KNN_classify_split_min_test(self, vectors, targets, known_task_id):
        # min_others test
        assert len(self._split_class_means)>0, 'split_class_means should not be empty!'
        unknown_scores = []
        known_scores = []
        task_begin, task_end = 0, 0
        task_id_targets = torch.zeros(vectors.shape[0], dtype=int).cuda()

        if known_task_id == 0:
            normed_vectors = F.normalize(vectors, dim=1)# 对特征向量做归一化
            dists = torch.cdist(normed_vectors, self._split_class_means[0], p=2)
            nme_scores = 1 - torch.softmax(dists, dim=-1)
            nme_max_scores, nme_preds = torch.max(nme_scores, dim=-1)
            return nme_preds, nme_max_scores, len(vectors)
        else:
            for task_id in range(known_task_id+1):
                task_end += self._increment_steps[task_id]

                task_head_features = vectors[:, task_id*self._feature_dim: (task_id+1)*self._feature_dim]
                normed_vectors = F.normalize(task_head_features, dim=1)# 对特征向量做归一化
                dists = torch.cdist(normed_vectors, self._split_class_means[task_id], p=2)
                task_scores = 1 - torch.softmax(dists, dim=-1)

                unknown_scores.append(task_scores[:, -1])

                known_scores_temp = torch.zeros((vectors.shape[0], max(self._increment_steps))).cuda()
                known_scores_temp[:, :(task_scores.shape[1]-1)] = task_scores[:, :-1]
                known_scores.append(known_scores_temp)

                # generate task_id_targets
                task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end)).squeeze(-1)
                if len(task_data_idxs) > 0:
                    task_id_targets[task_data_idxs] = task_id

                task_begin = task_end

            known_scores = torch.stack(known_scores, dim=0) # task num, b, max(task_sizes)
            unknown_scores = torch.stack(unknown_scores, dim=-1) # b, task num
            
            task_id_predict = torch.argmin(unknown_scores, dim=-1)
            task_id_correct = task_id_predict.eq(task_id_targets).cpu().sum()

            nme_preds = torch.zeros(vectors.shape[0], dtype=int).cuda()
            nme_max_scores = torch.zeros(vectors.shape[0]).cuda()
            task_begin, task_end = 0, 0
            for task_id in range(known_task_id+1):
                task_end += self._increment_steps[task_id] # do not have others category !
                task_logits_idxs = torch.argwhere(task_id_predict==task_id).squeeze(-1)
                if len(task_logits_idxs) > 0:
                    # nme_preds[task_logits_idxs] = torch.argmax(known_scores[id, task_logits_idxs], dim=1) + task_end
                    nme_max_scores[task_logits_idxs], nme_preds[task_logits_idxs] = torch.max(known_scores[task_id, task_logits_idxs], dim=1)
                    nme_preds[task_logits_idxs] += task_begin
                
                task_begin = task_end

            return nme_preds, nme_max_scores, task_id_correct