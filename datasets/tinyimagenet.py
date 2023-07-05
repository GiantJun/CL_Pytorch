from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np
import sys
from utils.toolkit import split_images_labels

class TinyImageNet(iData):

    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size if img_size != None else 64
        self.train_trsf = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63/255)
        ]
        self.test_trsf = [
        ]
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.230, 0.227, 0.226]),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        self.root_dir = os.path.join(os.environ["DATA"], "tiny-imagenet-200")
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        self._create_class_idx_dict_train()
        self._create_class_idx_dict_val()

        self._make_dataset()

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))

        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        train_images = []
        test_images = []
        train_list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        test_list_of_dirs = ["images"]

        for tgt in train_list_of_dirs:
            dirs = os.path.join(self.train_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        #add picture loading code
                        item = (path, self.class_to_tgt_idx[tgt])
                        train_images.append(item)
        
        for tgt in test_list_of_dirs:
            dirs = os.path.join(self.val_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        test_images.append(item)

        self.train_data, self.train_targets = split_images_labels(train_images)
        self.test_data, self.test_targets = split_images_labels(test_images)