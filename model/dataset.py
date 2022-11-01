import os
import glob
from unicodedata import category
import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils import data
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

image_dir = 'data/images/'
dirs_to_ignore = ['ShopeePay-Near-Me-cat', 'Miscellaneous-cat', 'Dining-Travel-Services-cat']
files_to_ignore = ['Automotive-cat\\9206333060.png', 'Men\'s-Bags-cat\\2391472522.png', 'Beauty-Personal-Care-cat\\157749.png', 'Beauty-Personal-Care-cat\\159257.png', 'Beauty-Personal-Care-cat\\2229429.png']
use_max_num_img = True
all_img_dim = (224,224)

class DataSet(data.Dataset):

    def __init__(self, path = None, max_num_img=None):
        self.data_dir = path if path else image_dir
        self.all_data_files = [f for f in glob.glob(self.data_dir + "**/*.png", recursive=True) if not self._ignore_file(f)]
        self.categories = [f.split('-cat')[0] for f in os.listdir(self.data_dir) if not f.startswith('.') and all(f not in dir for dir in dirs_to_ignore)]
        self.images = {}
        self.labels = {}
        self.loaded_files = []
        self.cat_map = dict(zip(self.categories, range(0, len(self.categories))))
        if max_num_img:
            self.max_num_img = max_num_img

    def __getitem__(self, idx):
        if len(self.loaded_files) == 0:
            return self.load_item_from_file(self.all_data_files[idx])
        
        key = self.loaded_files[idx]
        if key in self.images:
            return self.images[key], self.labels[key]

    def __len__(self):
        if len(self.loaded_files) == 0:
            return len(self.all_data_files)
        
        return len(self.loaded_files)

    def _ignore_file(self, file):
        if any(dir in file for dir in dirs_to_ignore):
            return True
        if any(file_ignore in file for file_ignore in files_to_ignore):
            return True
        return False

    def _get_categories(self):
        return [f.split('-cat')[0] for f in os.listdir(self.data_dir) if not f.startswith('.') and all(f not in dir for dir in dirs_to_ignore)]

    # def load_file(self, file):
    #     return read_image(file, mode=ImageReadMode.RGB)

    def image_count_per_category(self):
        counts = {}
        for cat in self.categories:
            counts[cat] = len(os.listdir(self.data_dir + cat + '-cat'))
        return(counts)

    def category_from_path(self, p):
        parts = p.split('/')
        dir = parts[-2]
        return dir.split('-cat')[0]

    def load_item_from_file(self, cat_file, cat_id=None):
        if cat_id is None:
            cat = self.category_from_path(cat_file)
            cat_id = self.categories.index(cat)
        img = read_image(cat_file, mode=ImageReadMode.RGB)
        self.images[cat_file] = self.preprocess_image(img)
        self.labels[cat_file] = cat_id
        return self.images[cat_file], self.labels[cat_file]

    def load_all(self):
        self.images = {}
        self.labels = {}
        self.loaded_files = []
        for cat_id in range(len(self.categories)):
            category = self.categories[cat_id]
            cat_files = [f for f in self.all_data_files if category in f]
           
            curr_max = self.max_num_img
            if curr_max is None:
                curr_max = len(cat_files)
            else:
                curr_max = min(len(cat_files), curr_max)

            for i in tqdm(range(curr_max)):
                try:
                    
                    #img = self.load_file(cat_files[i])
                    # img = cv2.imread(cat_files[i])
                    # img = cv2.resize(img, all_img_dim)
                    #if img.shape != all_img_dim:
                    #    print(img.shape)
                    #    img = cv2.resize(img, all_img_dim, interpolation=cv2.INTER_LINEAR)
                    # img = img/255.
                    # img = torch.tensor(img).permute(2,0,1)
                    # self.images.append(img.float())

                    self.loaded_files.append(cat_files[i])
                    self.load_item_from_file(cat_files[i], cat_id=cat_id)
                except Exception as e:
                    print('Error loading file: ' + cat_files[i], e)

    def plot_samples(self, category):
        cat_files = [f for f in self.all_data_files if category in f]
        paths = np.random.choice(cat_files, 4, replace=False)

        plt.figure(figsize=(12,12))
        for i in range(4):
            image = cv2.imread(paths[i])[...,[2,1,0]]
            image = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
            plt.subplot(1, 4, i+1)
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


    def preprocess_image(self, img, crop=0.75):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(all_img_dim),
            transforms.CenterCrop((int(crop*all_img_dim[0]), int(crop*all_img_dim[1]))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        img = transform(img)
        return img