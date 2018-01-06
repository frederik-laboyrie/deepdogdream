from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import numpy as np
import os
from PIL import Image

def preprocess_image(image_path, target_size=(256, 256), one_minus_one=False):
    img = load_img(image_path)
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    if one_minus_one:
        img -= 0.5
        img *= 2
    return img

def generate_array(directory_path):
    image_paths = [directory_path + path for path in os.listdir(directory_path) if '.jpg' in path]
    dog_list = [preprocess_image(image_path) for image_path in image_paths]
    dog_array = np.vstack(dog_list)
    return dog_array

def deprocess_image(x, one_minus_one=False):
    x = x.reshape((3, x.shape[2], x.shape[3]))
    x = x.transpose((1, 2, 0))
    x = x.reshape((x.shape[1], x.shape[2], 3))
    if one_minus_one:
        x /= 2.
        x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x
