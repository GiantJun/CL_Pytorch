from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np
from utils.toolkit import split_images_labels

class ImageNet_R(iData):
    '''
    Dataset Name:   ImageNet_R dataset
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
        self.use_path = True
        self.img_size = img_size if img_size != None else 224
        self.train_trsf = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63/255)
        ]
        self.test_trsf = []
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.230, 0.227, 0.226]),
        ]
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dataset = datasets.ImageFolder(os.path.join(os.environ['DATA'], 'imagenet_r', 'train'))
        test_dataset = datasets.ImageFolder(os.path.join(os.environ['DATA'], 'imagenet_r', 'test'))
        
        self.train_data, self.train_targets = split_images_labels(train_dataset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dataset.imgs)