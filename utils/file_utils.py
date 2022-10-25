import os
import glob
import skimage.io
from tqdm import tqdm

images_zip = 'data/images.tar.gz'
images = 'data/images'

def check_tar_exists(file = images_zip):
    if not os.path.exists(file):
        print('Zip file not found: %s' % file)
        return False
    return True

def extract_tar(file = images_zip):
    if not check_tar_exists(file):
        return
    print('Extracting %s' % file)
    os.system('tar -xzf %s -C data' % file)

def check_images_dir(dir = images):
    if not os.path.exists(dir):
        print('Imagse dir not found')
        return False
    return True

def prepare_images():
    if not check_images_dir():
        extract_tar()
    else:
        print('Images dir exists')

def load_images_skimage(path):
    ext = [".png"]
    files = []
    images = []
    [files.extend(glob.glob(path + "/" + folder + '/*' + ".png")) for folder in tqdm(os.listdir(path))]
    images.extend([skimage.io.imread(file) for file in tqdm(files)])
    return images