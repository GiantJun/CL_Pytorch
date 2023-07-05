from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np

class Skin8(iData):
    '''
    Dataset Name:   Skin8 (ISIC_2019_Classification)
    Task:           Skin disease classification
    Data Format:    600x450 color images.
    Data Amount:    3555 for training, 705 for validationg/testing
    Class Num:      8
    Notes:          balanced each sample num of each class

    Reference:      
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(224, (0.8, 1)),
            ]
        
        self.test_trsf = []
        
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.60298395, 0.4887822, 0.46266827], std=[0.25993535, 0.24081337, 0.24418062]),
        ]

        self.class_order = np.arange(8).tolist()
    
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
        base_dir = os.path.join(os.environ["DATA"], "ISIC2019")
        train_dir = os.path.join("datasets", "skin8_train_500.txt")
        test_dir = os.path.join("datasets", "skin8_test_500.txt")
        
        self.train_data, self.train_targets = self.getdata(train_dir, base_dir)
        self.test_data, self.test_targets = self.getdata(test_dir, base_dir)