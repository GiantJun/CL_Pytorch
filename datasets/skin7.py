from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np
import pandas as pd

class Skin7(iData):
    '''
    Dataset Name:   Skin7 (ISIC_2018_Classification)
    Task:           Skin disease classification
    Data Format:    600x450 color images.
    Data Amount:    8010 for training, 2005 for validationg/testing
    Class Num:      7
    Label:          

    Reference:      https://link.springer.com/chapter/10.1007/978-3-319-46466-4_13
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

        self.class_order = np.arange(7).tolist()

    def getdata(self, fn, img_dir):
        print(fn)
        csvfile = pd.read_csv(fn)
        raw_data = csvfile.values

        data = []
        targets = []
        for path, label in raw_data:
            data.append(os.path.join(img_dir, path))
            targets.append(label)

        return np.array(data), np.array(targets)

    def download_data(self):
        train_dir = os.path.join(os.environ["DATA"], "ISIC_2018_Classification/split_data/split_data_1_fold_train.csv")
        test_dir = os.path.join(os.environ["DATA"], "ISIC_2018_Classification/split_data/split_data_1_fold_test.csv")
        img_dir = os.path.join(os.environ["DATA"], "ISIC_2018_Classification/ISIC2018_Task3_Training_Input")

        self.train_data, self.train_targets = self.getdata(train_dir, img_dir)
        self.test_data, self.test_targets = self.getdata(test_dir, img_dir)