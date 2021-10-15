from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Office31(ImageList):
    """Office31 Dataset.

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """
    download_list = [
            ("image_list", "image_list.zip", "https://drive.google.com/uc?export=download&id=1JGjr1bYe0oYkso6prudvKYoFi0FOJcrE"),
            ("amazon", "amazon.tgz", "https://drive.google.com/uc?export=download&id=1xq7gPW14FSLlrerR9nDCryKeSBHNNeFp"),
            ("dslr", "dslr.tgz", "https://drive.google.com/uc?export=download&id=14F7HWvclPehy38aVMNNap1oFLVitUAgA"),
            ("webcam", "webcam.tgz", "https://drive.google.com/uc?export=download&id=11OW_6J7kss6nlgKmbX6jWWQbO3yTmHNV"),
    ]
    image_list = {
        "A": "image_list/amazon.txt",
        "D": "image_list/dslr.txt",
        "W": "image_list/webcam.txt"
    }
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Office31, self).__init__(root, Office31.CLASSES, data_list_file=data_list_file, **kwargs)
