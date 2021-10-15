import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import numpy as np

class ImageList(datasets.VisionDataset):
    """A generic Dataset class for domain adaptation in image classification

    Parameters:
        - **root** (str): Root directory of dataset
        - **classes** (List[str]): The names of all the classes
        - **data_list_file** (str): File to read the image list from.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride `parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str, 
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, subsample: Optional[bool] = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = self.parse_data_file(data_list_file)
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

        self.labels_to_idx = self.get_labels_to_idx(self.data)
        self.loader = default_loader

        if subsample:
            self.data = self.subsample(self.data, self.labels_to_idx)
            self.labels_to_idx = self.get_labels_to_idx(self.data)

        n = len(self.data)
        self.proportion = [len(self.labels_to_idx[key])/n for key in sorted(self.labels_to_idx.keys())]
        
        print(len(self.data))
    
    def subsample(self, data, labels_to_idx):
        np.random.seed(0)
        keep_idx = []
        num_classes = len(labels_to_idx)
        for label in sorted(labels_to_idx.keys()):
            if label < num_classes//2:
                keep_idx.extend(np.random.choice(labels_to_idx[label], int(0.3*len(labels_to_idx[label])), replace=False).tolist())
            else:
                keep_idx.extend(labels_to_idx[label])
        keep_idx = set(keep_idx)
        temp = []
        for i in range(len(data)):
            if i in keep_idx:
                temp.append(data[i])
        return temp
 
    def get_labels_to_idx(self, data):
        labels_to_idx = {}
        for idx, path in enumerate(data):
            label = path[1]
            if label not in labels_to_idx:
                labels_to_idx[label] = [idx]
            else:
                labels_to_idx[label].append(idx)
        return labels_to_idx


    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)
