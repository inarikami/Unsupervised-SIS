import keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from utilz import utils_data
import scipy.misc
from random import randint
cwd = os.getcwd()


class Autoencoder_Callbacks(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        # print(np.shape(self.validation_data[0][0]))
        return

    def on_epoch_end(self, epoch, logs=None):


        os.makedirs('images/', exist_ok=True)

        c = 2 # orig & reconstructed DO NOT CHANGE
        r = 5

        validation_length = len(self.validation_data[0])
        #first image for historical reference 
        img_array = []
        img_array.append(self.validation_data[0][0])
        #load image
        for x in range(r-1):
            img_array.append(self.validation_data[0][randint(0, validation_length-1)])
        
        np.concatenate(img_array)

        gen_img_array = []
        for img in img_array:
            gen_img_array.append(self.model.predict(np.array([img]))[0])

        np.concatenate(gen_img_array)

        original = []
        reconstruction = []
        for img in img_array:
            original.append(np.array(utils_data.denormalize(img)).astype('uint8'))
            # cv2.imshow("Simple_black", original)
            # cv2.waitKey(0)
        for gen_img in gen_img_array:
            reconstruction.append(np.array(utils_data.denormalize(gen_img)).astype('uint8'))
            # cv2.imshow("reconstruction", reconstruction)
            # cv2.waitKey(0)

        np.concatenate(original)
        np.concatenate(reconstruction)

        gen_imgs = [original, reconstruction]

        titles = ['Original', 'Reconstructed']

        fig, axs = plt.subplots(nrows=r,ncols=c)
        fig.set_size_inches(18.5, 10.5)
        
        for i in range(r): 
            for j in range(c):
                axs[i,j].imshow(gen_imgs[j][i])
                axs[i,j].set_title(titles[j])
                axs[i,j].axis('off')
        fig.savefig(cwd+"/images/%d.png" % (epoch),dpi=200)
        plt.close()

        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return


class DQN_Callbacks(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        # print(np.shape(self.validation_data[0][0]))
        return

    def on_epoch_end(self, epoch, logs=None):

        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return