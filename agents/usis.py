
from keras.layers import Lambda, Input, Conv2D, \
    UpSampling2D, Deconv2D, Concatenate, Dropout, Dense, Flatten, \
    ZeroPadding2D, AveragePooling2D, BatchNormalization, \
    DepthwiseConv2D, Activation, Reshape, Add, Conv2DTranspose, Softmax
from keras_contrib.layers.normalization import InstanceNormalization
from keras.engine import Layer, InputSpec
from keras import layers
from keras.utils import multi_gpu_model, plot_model, conv_utils
from keras.callbacks import TensorBoard
from keras.models import load_model, Model
from keras.losses import mean_squared_logarithmic_error, mse
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint


from utilz import utils_data
from agents import agent_callbacks
from agents import agent
import config
config = config.Config()

class USIS(agent.Agent):
    def __init__(self, classes = 20, filters = 64, img_cols = config.RESOLUTION_CAPTURE_WIDTH, 
    img_rows = config.RESOLUTION_CAPTURE_HEIGHT):
        super()
        self.classes = classes
        self.filters = filters
        self.img_cols = img_cols
        self.img_rows = img_rows
        self.img_shape = (self.img_rows, self.img_cols, 3) #3 for rgb
        self.batch_size = 8
        self.model = None


    def build_model(self):
        #builds the 'w' net

        inputs = Input(shape=self.img_shape)
        

        x = self.u_net(inputs, 'unet_encode_') #encoder

        x = Conv2D(self.classes, kernel_size=1, padding='same',
        use_bias=False, activation='softmax',
        name='expand_1')(x)

        x = Softmax(axis=-1)(x)

        x = self.u_net(x, 'unet_decode_') #decoder

        x = Conv2D(3, kernel_size=1, padding='same',
        use_bias=False, activation='relu',
        name='expand_3')(x)

        self.model = Model(inputs, x)
        self.model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])

        #self.usis.compile(optimizer='adam')
        plot_model(self.model, to_file="vae_plot.png", show_layer_names=True, show_shapes=True)


    def train_model(self):
        if self.model!= None:
            #do something
            #load data to memory from disk
                    #load testing and training data
            x_train, x_test = utils_data.load_images(config.DATA.get('train'),config.DATA.get('test'))  

            print(x_train.shape)
            print(x_test.shape)
            #tensorboard --logdir path_to_current_dir/Graph/ to see visual progress     
            tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True, update_freq='epoch')

            #custom callback to show learning progress
            callback = agent_callbacks.Autoencoder_Callbacks()

            # checkpoint callback
            checkpoint = ModelCheckpoint("weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

            callbacks_list = [checkpoint, callback, tbCallBack]

            self.model.fit(x_train, x_train,
                    epochs=100,
                    batch_size=8, 
                    shuffle=True,
                    validation_data=(x_test,x_test), 
                    callbacks=callbacks_list)



    def u_net(self, inputs, prefix, squeeze=4):
        x = inputs
        skip_connections = []

        for i in range(squeeze):
            x = Conv2D(self.filters*(i+1), kernel_size=1, padding='same',
                    use_bias=False, activation=None,
                    name=prefix + f'contract_Conv1_{i}')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                name=prefix + f'contract_BN1_{i}')(x)
            x = Activation(self.relu6, name=prefix + f'contract_relu1_{i}')(x)

            x = Conv2D(self.filters*(i+1), kernel_size=1, padding='same',
                    use_bias=False, activation=None,
                    name=prefix + f'contract_Conv2_{i}')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                name=prefix + f'contract_BN2_{i}')(x)
            x = Activation(self.relu6, name=prefix + f'contract_relu2_{i}')(x)

            skip_connections.append(x)

            x = Conv2D(self.filters*(i+1)*2, kernel_size=4, strides=2, padding='same',
                    use_bias=False, activation='relu',
                    name=prefix + f'contract_Conv3_{i}')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                    name=prefix + f'contract_BN3_{i}')(x)
        
        #expand out width
        x = Conv2D(self.filters*squeeze**2, kernel_size=1, padding='same',
                    use_bias=False, activation=None,
                    name=prefix + 'mid_1')(x)
        x = Conv2D(self.filters*squeeze**2, kernel_size=1, padding='same',
                    use_bias=False, activation=None,
                    name=prefix + 'mid_2')(x) 

        for i in range(squeeze):
            #up conv2x2
            x = Conv2DTranspose(self.filters*(squeeze-i), kernel_size=4 ,strides=2, 
                padding='same', activation='relu', name=prefix + f'expand_Conv1_{i}')(x)

            x = Concatenate(name= prefix + f'concat{i}', axis=-1)([skip_connections.pop(),x])

            x = Conv2D(self.filters*(squeeze-i), kernel_size=1, padding='same',
                use_bias=False, activation=None,
                name=prefix + f'expand_Conv2_{i}')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                name=prefix + f'expand_BN1_{i}')(x)
            x = Activation(self.relu6, name=prefix + f'expand_relu_1_{i}')(x)

            x = Conv2D(self.filters*(squeeze-i), kernel_size=1, padding='same',
                use_bias=False, activation=None,
                name=prefix + f'expand_Conv3_{i}')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                name=prefix + f'expand_BN2_{i}')(x)
            x = Activation(self.relu6, name=prefix + f'expand_relu_2_{i}')(x)
        return x



    def relu6(self, x):
        return K.relu(x, max_value=6)