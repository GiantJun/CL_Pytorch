import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


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

# def 

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
    def __init__(self, data, targets, transform, use_path=False, two_view=False):
        assert len(data) == len(targets), 'Data size error!'
        self.data = data
        self.targets = targets
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
        if self.two_view:
            if self.use_path:
                image2 = self.transform(pil_loader(self.data[idx]))
            else:
                image2 = self.transform(Image.fromarray(self.data[idx]))
            image = [image, image2]
            return idx, image, label
        else:
            return idx, image, label

def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')