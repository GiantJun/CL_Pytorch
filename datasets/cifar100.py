from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np

class CIFAR100(iData):
    '''
    Dataset Name:   CIFAR-100 dataset (Canadian Institute for Advanced Research, 100 classes)
    Source:         A subset of the Tiny Images dataset.
    Task:           Classification Task
    Data Format:    32x32 color images.
    Data Amount:    60000 (500 training images and 100 testing images per class)
    Class Num:      100 (grouped into 20 superclass).
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = False
        self.img_size = img_size if img_size != None else 32
        self.train_trsf = [
            transforms.RandomCrop(32, padding=4), # 32, 4
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=63/255)
        ]
        self.test_trsf = []
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(os.environ['DATA'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(os.environ['DATA'], train=False, download=True)
        
        # self.class_to_idx = train_dataset.class_to_idx
        self.class_to_idx = {}
        for key, value in train_dataset.class_to_idx.items():
            if key == 'aquarium_fish':
                print('class {}: {} => {}'.format(value, key, 'goldfish'))
                key = 'goldfish'
            elif key == 'lawn_mower':
                print('class {}: {} => {}'.format(value, key, 'mower'))
                key = 'mower'
            elif key == 'maple_tree':
                print('class {}: {} => {}'.format(value, key, 'maple'))
                key = 'maple'
            elif key == 'oak_tree':
                print('class {}: {} => {}'.format(value, key, 'oak'))
                key = 'oak'
            elif key == 'palm_tree':
                print('class {}: {} => {}'.format(value, key, 'palm'))
                key = 'palm'
            elif key == 'pickup_truck':
                print('class {}: {} => {}'.format(value, key, 'truck'))
                key = 'truck'
            elif key == 'pine_tree':
                print('class {}: {} => {}'.format(value, key, 'pine'))
                key = 'pine'
            elif key == 'sweet_pepper':
                print('class {}: {} => {}'.format(value, key, 'pepper'))
                key = 'pepper'
            elif key == 'willow_tree':
                print('class {}: {} => {}'.format(value, key, 'willow'))
                key = 'willow'
            else:
                print('class {}: {}'.format(value, key))
            self.class_to_idx[key] = value

        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)