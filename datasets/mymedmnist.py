from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np
import random

class MyMedMnist(iData):
    '''
    Dataset Name:   MedMNistv2
    Task:           Diverse classification task (binary/multi-class, ordinal regression and multi-label)
    Data Format:    32x32 color images.
    Data Amount:    Consists of 12 pre-processed 2D datasets and 6 pre-processed 3D datasets with diverse data scales (from 100 to 100,000)
    
    Reference: https://medmnist.com/
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        # 由于 MedMnist 中有些子数据集是多标签或者是3D的，这里没有使用
        # 以下表示中，字符为子数据集名字, 数字为子数据集中包含的类别数
        self.has_valid = True
        self._dataset_info = [('bloodmnist',8), ('organmnist_axial',11), ('pathmnist',9), ('tissuemnist',8)]
        # self._dataset_info = [('bloodmnist',8), ('organamnist',11), ('dermamnist',7), ('pneumoniamnist',2), ('pathmnist',9),
        #                     ('breastmnist',2), ('tissuemnist',8), ('octmnist',4)]
        self.use_path = False
        self.img_size = img_size if img_size != None else 28 # original img size is 28
        self.train_trsf = [
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.24705882352941178),
        ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.test_trsf = []
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]

        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]
        self.class_order = list(range(sum(self._dataset_inc)))
    
    def shuffle_order(self, seed):
        random.seed(seed)
        random.shuffle(self._dataset_info)
        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]

    def getdata(self, src_dir):
        train_data = np.array([])
        train_targets = np.array([])
        test_data = np.array([])
        test_targets = np.array([])
        
        known_class = 0
        for data_flag in self._dataset_info:
            npz_file = np.load(os.path.join(src_dir, "{}.npz".format(data_flag[0])))

            train_imgs = npz_file['train_images']
            train_labels = npz_file['train_labels']

            test_imgs = npz_file['test_images']
            test_labels = npz_file['test_labels']

            if len(train_imgs.shape) == 3:
                train_imgs = np.expand_dims(train_imgs, axis=3)
                train_imgs = np.repeat(train_imgs, 3, axis=3)

                test_imgs = np.expand_dims(test_imgs, axis=3)
                test_imgs = np.repeat(test_imgs, 3, axis=3)

            train_labels = train_labels + known_class
            train_imgs = train_imgs.astype(np.uint8)
            train_labels = train_labels.astype(np.uint8)

            test_labels = test_labels + known_class
            test_imgs = test_imgs.astype(np.uint8)
            test_labels = test_labels.astype(np.uint8)
            
            train_data = np.concatenate([train_data, train_imgs]) if len(train_data) != 0 else train_imgs
            train_targets = np.concatenate([train_targets, train_labels]) if len(train_targets) != 0 else train_labels
            

            test_data = np.concatenate([test_data, test_imgs]) if len(test_data) != 0 else test_imgs
            test_targets = np.concatenate([test_targets, test_labels]) if len(test_targets) != 0 else test_labels

            known_class = len(np.unique(train_targets))
        
        return train_data, np.squeeze(train_targets,1), test_data, np.squeeze(test_targets,1)


    def download_data(self):
        # 在我们划分的数据集中，后面这一项有几种选项可选
        # mymedmnist_1_in_5: 对不均衡的 PathMNIST,OCTMNIST, TissueMNIST, OrganAMNIST, 将其随机下采样为其原来的1/5, 其余不变
        # mymedmnist_1000_300: 对所有子数据集, 均随机采样成样本数分别为 1000,300 的训练集和测试集
        # origin: MedMNIST 原始数据集
        src_dir = os.path.join(os.environ["DATA"], "medmnist")
        self.train_data, self.train_targets, self.test_data, self.test_targets = self.getdata(src_dir)
        # print(self.train_data.shape)
        # print(self.test_data.shape)

        # print(len(np.unique(self.train_targets))) # output: 51
        # print(len(np.unique(self.test_targets))) # output: 51