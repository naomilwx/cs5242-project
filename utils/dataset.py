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

image_dir = 'data/images/'
dirs_to_ignore = ['ShopeePay-Near-Me', 'Miscellaneous']
files_to_ignore = ['Automotive-cat\\9206333060.png', 'Men\'s-Bags-cat\\2391472522.png']
resized_img_dim = (512,512)

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
            for i in tqdm(range(len(cat_files))):

                try:
                    
                    #img = self.load_file(cat_files[i])
                    #img = read_image(cat_files[i])
                    img = cv2.imread(cat_files[i])
                    img_resized = cv2.resize(img, resized_img_dim, interpolation=cv2.INTER_LINEAR)
                    self.images.append(img_resized)
                    self.labels.append(cat_id)
                except:
                    print('Error loading file: ' + cat_files[i])

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