import os
import glob
import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import PIL

from torch.utils import data
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision import transforms

image_dir = 'data/images/'

all_categories = ["Men's-Bags", 'Jewellery-Accessories', 'Automotive', "Men's-Shoes", 'Computers-Peripherals', 'Beauty-Personal-Care', 'Home-Appliances', 'Pet-Food-Supplies', 'Food-Beverages', "Women's-Apparel", 'Kids-Fashion', 'Video-Games', 'Mobile-Gadgets', 'Watches', 'Cameras-Drones', 'Travel-Luggage', 'Sports-Outdoors', 'Toys-Kids-Babies', 'Home-Living', 'Hobbies-Books', "Men's-Wear", 'Health-Wellness', "Women's-Bags", "Women's-Shoes"]
categories_to_include = ["Men's-Bags", "Women's-Bags", "Women's-Apparel", "Men's-Wear", 'Kids-Fashion', "Men's-Shoes", "Women's-Shoes", 'Jewellery-Accessories', 'Watches']
dirs_to_ignore = ['ShopeePay-Near-Me-cat', 'Miscellaneous-cat', 'Dining-Travel-Services-cat']
files_to_ignore = [
    # Automotive-cat
    '9206333060.png',
    # Men\'s-Bags-cat
    '2391472522.png',
    # Beauty-Personal-Care-cat
    '157749.png',
    '159257.png',
    '2229429.png',
    # Video-Games
    '14441529.png',
    '70096765.png',
    '1996160760.png',
    # 'Toys-Kids-Babies'
    '1150552610.png',
    # 'Women\'s-Apparel'
    '5920595313.png'
]
use_max_num_img = True
all_img_dim = (224,224)

class DataSet(data.Dataset):
    def __init__(self, path = None, max_num_img=None, crop=0.75, categories=categories_to_include):
        self.data_dir = path if path else image_dir
        self.categories = categories
        self.all_data_files = [f for f in glob.glob(self.data_dir + "**/*.png", recursive=True) if not self._ignore_file(f)]
        # self.categories = [f.split('-cat')[0] for f in os.listdir(self.data_dir) if not f.startswith('.') and all(f not in dir for dir in dirs_to_ignore)]
        self.images = {}
        self.labels = {}
        self.loaded_files = []
        self.cat_map = dict(zip(self.categories, range(0, len(self.categories))))
        if max_num_img:
            self.max_num_img = max_num_img
        self.crop = crop

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
        dir, filename = os.path.split(file)
        # if any(folder in dir for folder in dirs_to_ignore):
        #     return True
        included_cat = any(cat in dir for cat in self.categories)
        if any(filename == file_ignore for file_ignore in files_to_ignore):
            return True
        return not included_cat

    # def _get_categories(self):
    #     return [f.split('-cat')[0] for f in os.listdir(self.data_dir) if not f.startswith('.') and all(f not in dir for dir in dirs_to_ignore)]

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
        img = PIL.Image.open(cat_file)
        img = transforms.functional.to_tensor(img)
        # img = read_image(cat_file, mode=ImageReadMode.RGB)
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
                    self.load_item_from_file(cat_files[i], cat_id=cat_id)
                    self.loaded_files.append(cat_files[i])
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

    def preprocess_image(self, img):
        crop = self.crop
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((int(crop*img.shape[1]), int(crop*img.shape[2]))),
            transforms.Resize(all_img_dim),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        img = transform(img)
        return img