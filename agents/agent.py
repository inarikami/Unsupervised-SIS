from keras.models import load_model, Model
from keras import backend as K
import numpy as np

class Agent(object):

    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.model = None
        pass

    def load_model(self, save_path):
        """
            Loads a saved compiled model.

        """
        if(self.model == None):
            self.model = load_model(save_path)
            return
        print("errs")
        return    

    def reset_model(self):
        """
            Resets the current model in memory.

        """
        if(self.model != None):
            self.model = None

    def get_model_memory_usage(self):
        shapes_mem_count = 0
        for l in self.model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(self.model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)])

        total_memory = 4.0*self.batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes

