import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from argparse import ArgumentParser

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy, count_parameters
from backbone.inc_net import IncrementalNet

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--K', type=int, default=None, help='K for lucir / memory bank size for moco / component for gmm_bayes')
    return parser

class GMM_Bayes(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        if self._incre_type != 'cil':
            raise ValueError('gmm_bayes is a class incremental method!')
        self._K = config.K
        self._gmm_bayes_classifier = None
        self._train_sample_num = None

    def prepare_task_data(self, data_manager):
        super().prepare_task_data(data_manager)
        if self._train_sample_num is None:
            self._train_sample_num = data_manager.train_sample_num

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
            self._network.freeze_FE()

            self._gmm_bayes_classifier = NaiveBayes(self._logger, self._train_sample_num, self._K)

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
    
    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, None, None, task_id=self._cur_task, epochs=1)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        feature_all, target_all = [], []
        model.train()

        for _, inputs, targets in train_loader:
            inputs= inputs.cuda()

            features, _ = model(inputs)

            feature_all.append(features.cpu())
            target_all.append(targets.numpy())
        
        feature_all = F.normalize(torch.cat(feature_all), dim=1).numpy()
        target_all = np.concatenate(target_all, 0)

        self._gmm_bayes_classifier.fit_new_task_data(feature_all, target_all)

        train_acc = 100.0
        train_loss = ['Loss', 0]
        return model, train_acc, train_loss

    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        cnn_pred_all, nme_pred_all, target_all, features_all = [], [], [], []
        cnn_max_scores_all, nme_max_scores_all = [], []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            if self._incre_type == 'cil':
                cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits[:,:task_end], dim=-1), dim=-1)
            elif self._incre_type == 'til':
                cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits[:, task_begin:task_end], dim=-1), dim=-1)
                cnn_preds = cnn_preds + task_begin
                
            if ret_pred_target:
                if self._memory_bank != None and self._apply_nme:
                    nme_pred, nme_max_scores = self._memory_bank.KNN_classify(feature_outputs['features'])
                    nme_pred_all.append(tensor2numpy(nme_pred))
                    nme_max_scores_all.append(tensor2numpy(nme_max_scores))
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                features_all.append(tensor2numpy(feature_outputs['features']))
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
            nme_pred_all = np.concatenate(nme_pred_all) if len(nme_pred_all) != 0 else nme_pred_all
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            nme_max_scores_all = np.concatenate(nme_max_scores_all) if len(nme_max_scores_all) !=0 else nme_max_scores_all
            target_all = np.concatenate(target_all)
            features_all = np.concatenate(features_all)
            return cnn_pred_all, nme_pred_all, cnn_max_scores_all, nme_max_scores_all, target_all, features_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                return test_acc, test_task_acc
            else:
                return test_acc


class NaiveBayes():
    # 注：这是朴素贝叶斯，而不是高斯判别模型
    def __init__(self, logger, train_sample_num, component=2) -> None:
        self._logger = logger
        self._train_sample_num = train_sample_num
        self._component = component
        self._class_gmm_list = []
        self._class_prior = []
    
    def fit_new_task_data(self, x, y):
        class_range = np.unique(y)
        for class_id in class_range:
            class_data = x[np.where(y==class_id)]
            self._class_gmm_list.append(self._fit_new_class_gmm(class_data))
            self._class_prior.append(class_data.shape[0] / self._train_sample_num)
            self._logger.info("Fitted class {} 's gaussian mixture model, prior={}".format(class_id, self._class_prior[-1]))
    
    def _fit_new_class_gmm(self, x):
        gmm_list = []
        # 将X转置,行代表通道数据
        # 对每个特征通道拟合一个高斯模型
        x_T = x.T # [b, feature_dim] => [feature_dim, b]
        for idx in range(x_T.shape[0]):
            gmm = GaussianMixture(n_components=self._component).fit(x_T[idx].reshape(-1, 1))
            gmm_list.append(gmm)        
        return gmm_list

    def predict(self, X):
        # 取概率最大的类别返回预测值
        # 计算每个种类的概率P(Y|x1,x2,x3) =  P(Y)*P(x1|Y)*P(x2|Y)*P(x3|Y)
        output = []
        for class_id in range(self._class_gmm_list):
            prior = self._class_prior[class_id]
            # posterior = self._pdf(X, class_id)
            posterior = self._gmm_pdf(X, class_id)
            prediction = prior + posterior
            output.append(prediction)
        output = np.reshape(output, (len(self._class_gmm_list), X.shape[0]))
        self.out_put_matrix = output
        prediction = np.argmax(output, axis=0)
        return prediction

    def _gmm_pdf(self, X, class_id):
        X_T = X.T # [b, feature_dim] => [feature_dim, b]
        gmm_list = self._class_gmm_list[class_id]
        result = np.zeros(X_T.shape)
        for idx in range(X_T.shape[0]):
            gmm = gmm_list[idx]
            pp = gmm.score_samples(X_T[idx].reshape(-1, 1))
            result[idx] = pp
        result = np.sum(result, axis=0) # [b]
        return result
    