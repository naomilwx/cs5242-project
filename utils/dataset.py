import os
import glob
import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torchvision.io import read_image
from torchvision import transforms, models, datasets

image_dir = 'data/images/'
dirs_to_ignore = ['ShopeePay-Near-Me-cat', 'Miscellaneous-cat']
files_to_ignore = ['Automotive-cat\\9206333060.png', 'Men\'s-Bags-cat\\2391472522.png', 'Beauty-Personal-Care-cat\\157749.png', 'Beauty-Personal-Care-cat\\159257.png', 'Beauty-Personal-Care-cat\\2229429.png']
use_max_num_img = True
max_num_img = 100
all_img_dim = (224,224)

class DataSet(data.Dataset):

    def __init__(self, path = None):
        self.data_dir = path if path else image_dir
        self.data_files = [f for f in glob.glob(self.data_dir + "**/*.png", recursive=True) if not self._ignore_file(f)]
        self.categories = [f.split('-cat')[0] for f in os.listdir(self.data_dir) if not f.startswith('.') and all(f not in dir for dir in dirs_to_ignore)]
        self.images = []
        self.labels = []
        self.cat_map = dict(zip(self.categories, range(0, len(self.categories))))

    def __getitem__(self, idx):
        return self.load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)

    def _ignore_file(self, file):
        if any(dir in file for dir in dirs_to_ignore):
            return True
        if any(file_ignore in file for file_ignore in files_to_ignore):
            return True
        return False

    def _get_categories(self):
        return [f.split('-cat')[0] for f in os.listdir(self.data_dir) if not f.startswith('.') and all(f not in dir for dir in dirs_to_ignore)]

    def load_file(self, file):
        return read_image(file)

    def image_count_per_category(self):
        counts = {}
        for cat in self.categories:
            counts[cat] = len(os.listdir(self.data_dir + cat + '-cat'))
        return(counts)

    def load_all(self):
        

        for cat_id in range(len(self.categories)):
            category = self.categories[cat_id]
            cat_files = [f for f in self.data_files if category in f]
            #if ~use_max_num_img:
            #    max_num_img = len(cat_files)
            if len(cat_files) ==0:
                curr_max = 0
            else:
                curr_max = max_num_img
            #for i in tqdm(range(len(cat_files))):
            for i in tqdm(range(curr_max)):

                try:
                    
                    #img = self.load_file(cat_files[i])
                    #img = read_image(cat_files[i])

                    img = cv2.imread(cat_files[i])
                    # img = cv2.resize(img, all_img_dim)
                    #if img.shape != all_img_dim:
                    #    print(img.shape)
                    #    img = cv2.resize(img, all_img_dim, interpolation=cv2.INTER_LINEAR)
                    # img = img/255.
                    # img = torch.tensor(img).permute(2,0,1)
                    # self.images.append(img.float())

                    self.images.append(self.preprocess_image(img))
                    self.labels.append(cat_id)
                except Exception as e:
                    print('Error loading file: ' + cat_files[i], e)

    def plot_samples(self, category):
        cat_files = [f for f in self.data_files if category in f]
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
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(all_img_dim),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        img = transform(img)
        return img