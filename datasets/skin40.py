from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np

class Skin40(iData):
    '''
    Dataset Name:   Skin40 (Subset of SD-198)
    Task:           Skin disease classification
    Data Format:    224x224 color images. (origin imgs have different w,h)
    Data Amount:    2000 imgs for training and 400 for validationg/testing
    Class Num:      40
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

        self.class_order = np.arange(40).tolist()

    def getdata(self, fn, img_dir):
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(os.path.join(img_dir, temp[0]))
                targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        train_dir = os.path.join(os.environ["DATA"], "SD-198/main_classes_split/train_1.txt")
        test_dir = os.path.join(os.environ["DATA"], "SD-198/main_classes_split/val_1.txt")
        img_dir = os.path.join(os.environ["DATA"], 'SD-198/images')

        self.train_data, self.train_targets = self.getdata(train_dir, img_dir)
        self.test_data, self.test_targets = self.getdata(test_dir, img_dir)

        print(len(np.unique(self.train_targets))) # output: 
        print(len(np.unique(self.test_targets))) # output: 