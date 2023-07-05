from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np

class CIFAR10(iData):
    '''
    Dataset Name:   CIFAR-10 dataset (Canadian Institute for Advanced Research, 10 classes)
    Source:         A subset of the Tiny Images dataset.
    Task:           Classification Task
    Data Format:    32x32 color images.
    Data Amount:    60000 (5000 training images and 1000 testing images per class)
    Class Num:      10 (grouped into 2 superclass).
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
    
    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    def __init__(self, img_size:int=None) -> None:
        super().__init__()
        self.use_path = False
        self.img_size = img_size if img_size != None else 32
        self.train_trsf = [
            transforms.RandomCrop(32, padding=4), #32, 4
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=63/255)
        ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
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
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2615)),
        ]

        self.class_order = np.arange(10).tolist()

    def download_data(self):
        # or replay os.environ['xxx'] with './data/'
        train_dataset = datasets.cifar.CIFAR10(os.environ['DATA'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(os.environ['DATA'], train=False, download=True)
        
        self.class_to_idx = train_dataset.class_to_idx
        
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)