import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if hasattr(x, 'is_cuda') and x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).cuda()
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def check_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment):
    assert len(y_pred) == len(y_true), 'Data length error.'
    total_acc = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)
    known_classes = 0
    task_acc_list = []

    # Grouped accuracy
    for cur_classes in increment:
        idxes = np.where(np.logical_and(y_true >= known_classes, y_true < known_classes + cur_classes))[0]
        task_acc_list.append(np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2))
        known_classes += cur_classes
        if known_classes >= nb_old:
            break

    return total_acc, task_acc_list


def cal_openset_test_metrics(pred_max_score, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, pred_max_score)
    roc_auc = auc(fpr, tpr)
    fpr95_idx = np.where(tpr>=0.95)[0]
    fpr95 = fpr[fpr95_idx[0]]

    ap = average_precision_score(y_true, pred_max_score)
    return roc_auc*100, fpr95*100, ap*100


def cal_ece(y_pred, pred_max_score, y_true, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = (y_pred==y_true)
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # calculate |confidence - accuracy| in each bin
        in_bin = (pred_max_score>bin_lower.item()) * (pred_max_score <= bin_upper.item())
        prop_in_bin = in_bin.astype(float).mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = pred_max_score[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def cal_class_avg_acc(y_pred, y_true):
    class_acc_list = []
    for class_id in np.unique(y_true):
        class_mask = np.where(y_true==class_id)[0]
        class_acc_list.append((y_pred[class_mask]==class_id).sum() / len(class_mask))
    return np.mean(class_acc_list)*100

def mean_class_recall(y_pred, y_true, nb_old, increment):
    assert len(y_pred) == len(y_true), 'Data length error.'
    total_mcr = cal_mean_class_recall(y_pred, y_true)
    known_classes = 0
    task_mcr_list = []

    # Grouped accuracy
    for cur_classes in increment:
        idxes = np.where(np.logical_and(y_true >= known_classes, y_true < known_classes + cur_classes))[0]
        task_mcr_list.append(cal_mean_class_recall(y_pred[idxes], y_true[idxes]))
        known_classes += cur_classes
        if known_classes >= nb_old:
            break
    return total_mcr, task_mcr_list

def cal_mean_class_recall(y_pred, y_true):
    """ Calculate the mean class recall for the dataset X """
    cm = confusion_matrix(y_true, y_pred)
    right_of_class = np.diag(cm)
    num_of_class = cm.sum(axis=1)
    mcr = np.around((right_of_class*100 / (num_of_class+1e-8)).mean(), decimals=2)
    return mcr

def cal_bwf(task_metric_curve, cur_task):
    """cur_task in [0, T-1]"""
    bwf = 0.
    if cur_task > 0:
        for i in range(cur_task):
            task_result = 0.
            for j in range(cur_task-i):
                task_result += task_metric_curve[i][cur_task - j] - task_metric_curve[i][i]
            bwf += task_result / (cur_task-i)
        bwf /= cur_task
    return bwf

def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

class DummyDataset(Dataset):
    def __init__(self, data, targets, transform, use_path=False, two_view=False, soft_targets=None):
        assert len(data) == len(targets), 'Data size error!'
        self.data = data
        self.targets = targets
        self.soft_targets = soft_targets

        self.transform = transform
        self.use_path = use_path
        self.two_view = two_view

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.transform(pil_loader(self.data[idx]))
        else:
            image = self.transform(Image.fromarray(self.data[idx]))
        label = self.targets[idx]
        logit = self.soft_targets[idx] if self.soft_targets is not None else -1
        if self.two_view:
            if self.use_path:
                image2 = self.transform(pil_loader(self.data[idx]))
            else:
                image2 = self.transform(Image.fromarray(self.data[idx]))
            image = [image, image2]
            return logit, image, label
        else:
            return logit, image, label

def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')