import numpy as np
import os
cwd = os.getcwd()

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    RESOLUTION_GAME_WIDTH = 1024 
    RESOLUTION_GAME_HEIGHT = 768 

    RESOLUTION_CAPTURE_WIDTH = 224
    RESOLUTION_CAPTURE_HEIGHT = 224



##############################################
    MODEL_DIR = cwd+'/models/'

    DATA = {
        'train':cwd+'/data/train/',
        'test':cwd+'/data/test/'
        #TODO: Add saved data locations
        
    }


    PATH_AUTOENCODER_TEST = cwd+'/datasets/vae/test/'

##############################################

    GPU_COUNT = 1    # NUMBER OF GPUs to use





    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

if __name__ == '__main__':
    c = Config()
    c.display()
