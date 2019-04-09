import scipy
import numpy as np
import h5py
import os
from PIL import Image
from shutil import copyfile
from skimage.transform import resize
import glob
import config
from keras.applications import imagenet_utils
import skimage
import cv2
config = config.Config()


def load_h5(train_path, test_path):
    """ Load data for the vae autoencoder
    """
    image_data_train = []
    image_data_test = []

    image_data = [image_data_train, image_data_test]
    image_path = [train_path, test_path]

    for i in range(len(image_data)):
        #convert all h5py files to keras-readable array
        for h5file in os.listdir(image_path[i]):
            if h5file.endswith(".h5"):
                f = h5py.File(image_path[i]+h5file, 'r')
                image_data[i].append(f['Images'])
        image_data[i] = np.concatenate(image_data[i])

        if image_data[i].shape[1] != config.RESOLUTION_CAPTURE_HEIGHT or image_data[i].shape[2] != config.RESOLUTION_CAPTURE_WIDTH:
                raise  Exception('ERROR: Inconsistencies with training image shapes. Aborting Session')

    return normalize(image_data[0]), normalize(image_data[1])

def load_images(train_path, test_path):
    image_data_train = []
    image_data_test = []    # i wrote this when i was really tired pls revise and fix

    image_data = [image_data_train, image_data_test]
    image_path = [train_path, test_path]
    for i in range(len(image_data)):
        for imgfile in os.listdir(image_path[i]):
            if imgfile.endswith(".png") or imgfile.endswith(".jpg"):
                img = cv2.imread(image_path[i]+imgfile)
                if (img.shape[0] != config.RESOLUTION_CAPTURE_HEIGHT or img.shape[1] != config.RESOLUTION_CAPTURE_HEIGHT):
                    img = cv2.resize(img, dsize=(config.RESOLUTION_CAPTURE_HEIGHT, config.RESOLUTION_CAPTURE_WIDTH), interpolation=cv2.INTER_CUBIC)
                image_data[i].append(np.array(img))
    # print(image_data[0].shape)
    
    return normalize(np.array(image_data[0])), normalize(np.array(image_data[1]))

def prepare_images(data_path, name):
    #TODO: move all images in folders to one folder with ordered images and file names
    
    newpath = f'{data_path}{name}/'
    #create new folder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    i = 0
    for folder in os.walk(data_path):
        #organize images in folder
        for image in sorted(glob.glob(f'{data_path}{folder}/*.png')):  
            #add each image to new folder
            copyfile(data_path+image, f'{newpath}{i}.png')
            i+=1
    return


def crop_center(img, cropx, cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)


def normalize(x):
    """Normlaizes a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return (x - 128.0) / 128

def denormalize(x):
    """Denormalizes a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array scaled to [-1.,1.]
    # Returns
        Input array scaled to RGB values within [0, 255]
    """
    return round((x + 1) * 255 / 2)
    
def resize_input(x, height, width):
    images_resized = []
    length = x.shape[0]
    for i in range(length):
        original = x[i]
        resized = resize(original, (height,width), anti_aliasing=True)
        images_resized.append(resized)
        print(f"Processed {i}/{length}", end="\r")
    print(np.array(images_resized).shape)
    return np.array(images_resized)
