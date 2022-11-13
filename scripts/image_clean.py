import cv2
import math
import numpy as np
import imagehash
from PIL import Image
import os

try:
    import keras_ocr
except ImportError as e:
    print('keras-ocr not installed')
except Exception as e:
    print(e)

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def crop_image(img, scale):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def crop_and_inpaint_text(img_path, pipeline, crop_scale=0.8):
    # read image
    img = keras_ocr.tools.read(img_path)
    img = crop_image(img, crop_scale)
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(img)

def get_similar_images(dir):
    imghashes = {}
   
    for fn in os.listdir(dir):
        if fn[0] == '.':
            continue
        fp = os.path.join(dir, fn)
        if not os.path.isfile(fp):
            continue
        curr_hash =imagehash.average_hash(Image.open(fp))
        if curr_hash not in imghashes:
            imghashes[curr_hash] = []
        imghashes[curr_hash].append(fp)

    similar_imgs = []
    for _, imgs in imghashes.items():
        if len(imgs) > 1:
            similar_imgs.append(imgs)
    return similar_imgs

def get_duplicate_images(dir):
    imghashes = {}
   
    for fn in os.listdir(dir):
        if fn[0] == '.':
            continue
        fp = os.path.join(dir, fn)
        if not os.path.isfile(fp):
            continue
        curr_hash =imagehash.average_hash(Image.open(fp))
        if curr_hash not in imghashes:
            imghashes[curr_hash] = []
        imghashes[curr_hash].append(fp)

    dup_images = []
    for _, imgs in imghashes.items():
        if len(imgs) > 1:
            dups = get_duplicate_pairs(imgs)
            if len(dups) > 0:
                dup_images.extend(dups)
        
    return dup_images

def are_images_the_same(imgpath1, imgpath2):
    img1 = cv2.imread(imgpath1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_COLOR)
    if img1.shape != img2.shape:
        return False
    difference = cv2.subtract(img1, img2)
    b, g, r = cv2.split(difference)
    return cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0

def get_duplicate_pairs(images):
    dups = []
    for i in range(len(images)):
        for j in range(i+2, len(images)):
            if are_images_the_same(images[i], images[j]):
                dups.append((images[i], images[j]))

    return dups

def remove_background(data_dir, out_dir, target_folders):
    for folder in target_folders:
        if folder.startswith('.'):
            continue
        out_path = os.path.join(out_dir, folder)
    
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        print('processing', folder)
        cmd = f"rembg p \"{os.path.join(data_dir, folder)}\" \"{out_path}\""
        os.system(cmd)