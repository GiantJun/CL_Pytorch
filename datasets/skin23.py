from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np

class Skin23(iData):
    '''
    Dataset Name:   Skin23
    Task:           skin disease classification
    Data Format:    224x224 color images. (origin imgs have different w,h)
    Data Amount:    15,500 images for training and 4,000 for validationg/testing
    Class Num:      23
    Label:          

    Reference:      https://www.kaggle.com/datasets/shubhamgoel27/dermnet
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomHorizontalFlip(),
            ]
        self.test_trsf = []
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.60298395, 0.4887822, 0.46266827], std=[0.25993535, 0.24081337, 0.24418062]),
        ]

        self.class_order = np.arange(23).tolist()

    def getdata(self, img_dir):
        data, targets = [], []
        for class_id, class_name in enumerate(os.listdir(img_dir)):
            for img_id in os.listdir(os.path.join(img_dir, class_name)):
                data.append(os.path.join(img_dir, class_name, img_id))
                targets.append(class_id)
            
        return np.array(data), np.array(targets)

    def download_data(self):
        root_dir = os.path.join(os.environ["DATA"], 'skin23')
        self.train_data, self.train_targets = self.getdata(os.path.join(root_dir, 'train'))
        self.test_data, self.test_targets = self.getdata(os.path.join(root_dir, 'test'))