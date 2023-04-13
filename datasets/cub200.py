from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np

class CUB200(iData):
    '''
    Dataset Name:   CUB200-2011
    Task:           fine-grain birds classification
    Data Format:    224x224 color images. (origin imgs have different w,h)
    Data Amount:    5,994 images for training and 5,794 for validationg/testing
    Class Num:      200
    Label:          

    Reference:      https://opendatalab.com/CUB-200-2011
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomResizedCrop(224, (0.6, 1)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            ]
        
        self.test_trsf = []
        
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(200).tolist()

    def getdata(self, train:bool, img_dir):
        data, targets = [], []
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_train = line.split()
                if int(is_train) == int(train):
                    data.append(os.path.join(img_dir, self.images_path[image_id]))
                    targets.append(self.class_ids[image_id])
            
        return np.array(data), np.array(targets)

    def download_data(self):
        root_dir = os.path.join(os.environ["DATA"], 'CUB_200_2011')
        img_dir = os.path.join(root_dir, 'images')

        self.images_path = {}
        with open(os.path.join(root_dir, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(root_dir, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id

        self.train_data, self.train_targets = self.getdata(True, img_dir)
        self.test_data, self.test_targets = self.getdata(False, img_dir)

        print(len(np.unique(self.train_targets))) # output: 
        print(len(np.unique(self.test_targets))) # output: 