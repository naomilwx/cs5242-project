import os
import re
from model.dataset import DataSet as BaseDataset, categories_to_include

def remove_special_chars(str):
    return re.sub(r'[^\x00-\x7F]+', '', str)

class DataSet(BaseDataset):
    def __init__(self, product_names, path = None, max_num_img=None, crop=0.75, categories=categories_to_include):
        super(DataSet, self).__init__(path, max_num_img, crop, categories)
        self.product_names = product_names

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        fp = ''
        if len(self.loaded_files) == 0:
            fp = self.all_data_files[idx]
        else:
            fp = self.loaded_files[idx]
        _, filename = os.path.split(fp)
        pid = filename.split('.')[0]

        return image, remove_special_chars(self.product_names[pid]), label