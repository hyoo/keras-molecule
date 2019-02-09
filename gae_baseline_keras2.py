from __future__ import print_function

import argparse
import os
import h5py
import numpy as np


MODEL_SAVE = 'model.latent_dim.200'
DATA = 'prot.ohe.h5'

# num_epochs = NUM_EPOCHS
# batch_size = BATCH_SIZE
# latent_dim = LATENT_DIM
# random_seed = RANDOM_SEED
# model_save = MODEL_SAVE
# data = DATA


# add these to work in notebook w/o command line args
# num_epochs = args.epochs
# batch_size = args.batch_size
# latent_dim = args.latent_dim
# random_seed = args.random_seed
# model = args.model


# This is the candle stuff
import candle_keras as candle
import default_utils

additional_definitions = [{'name':'activation','type':'string'},
			{'name':'filter','type':'int'},
			{'name':'kernel_size','type':'int'}]

required = None
class gae(candle.Benchmark):  # 1
    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


# thread optimization
import os
from keras import backend as K
if K.backend() == 'tensorflow' and 'NUM_INTRA_THREADS' in os.environ:
    import tensorflow as tf
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
                                            intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))
    K.set_session(sess)



# this is a candle requirement
def initialize_parameters():
    gae_common = candle.Benchmark('./',
                    'gae_params.txt',
                    'keras',
                     prog='gae_baseline_keras2',
                     desc='GAE Network'
                 )

    # Initialize parameters
    gParameters = default_utils.initialize_parameters(gae_common)

    return gParameters






# This comes from the original work
# From molecules.model
import copy
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

class MoleculeVAE():

    autoencoder = None
    encoder = None
    decoder = None
    
    # TODO: find out what default learning rate should be
    def create(self, charset, max_length = 120,
               latent_rep_size = 292, weights_file = None,
	       optimizer='Adam', activation='relu',
               learning_rate=0.001):
        
        charset_length = len(charset)

        # Create a keras optimizer
        kerasDefaults = candle.keras_default_config()
        # This next line should be be dynamically set based on gParameters
        # kerasDefaults['momentum_sgd'] = gParameters['momentum']
        k_optimizer = candle.build_optimizer(optimizer,
                                        learning_rate,
                                        kerasDefaults)
        
        # Build the encoder
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length, activation=activation)
        self.encoder = Model(x, z)

        # Build the decoder
        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        # Build the autoencoder (encoder + decoder)
        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                charset_length,
                activation=activation
            )
        )

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        print("compiling autoencoder with optimizer = ", k_optimizer)
        self.autoencoder.compile(optimizer = k_optimizer,
                                 loss = vae_loss,
                                 metrics = ['accuracy'])

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01, activation='relu',filter=9, kernel_size=9):
        # Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
        # filter = 9
        # The length of the 1D convolution window
        # kernel_size = 9
        
        h = Convolution1D(filter, kernel_size, activation = activation, name='conv_1')(x)
        h = Convolution1D(filter, kernel_size, activation = activation, name='conv_2')(h)
        h = Convolution1D(10, 11, activation = activation, name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = activation, name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            # epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., std = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length, activation='relu'):
        h = Dense(latent_rep_size, name='latent_input')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 292):
        self.create(charset, weights_file = weights_file, latent_rep_size = latent_rep_size)



# From molecules.util
def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)



# globals to factor out
MODEL_SAVE = 'model.latent_dim.200'
DATA = 'prot.ohe.h5'

def run(gParameters):
    # args = get_arguments()

    num_epochs = gParameters['num_epochs']
    batch_size = gParameters['batch_size']
    latent_dim = gParameters['latent_dim']
    random_seed = gParameters['random_seed']
    activation = gParameters['activation']
    optimizer = gParameters['optimizer']
    model_save = MODEL_SAVE
    data = DATA



    np.random.seed(random_seed)
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    
    # load the data and create the autoencoder.
    print ('loading data: ', data)
    data_train, data_test, charset = load_dataset(data)
    model = MoleculeVAE()
    
    if os.path.isfile(model_save):
        model.load(charset, model_save, latent_rep_size = latent_dim)
    else:
        print("calling model.create with optimizer = ", optimizer)
        model.create(charset, latent_rep_size = latent_dim, activation = activation,
                    optimizer = optimizer )

    # create callbacks to be executed after each epoch.
    checkpointer = ModelCheckpoint(filepath = model_save,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    if(True):
        model.autoencoder.summary()
        model.autoencoder.fit(
            data_train,
            data_train,
            shuffle = True,
            epochs = num_epochs,
            batch_size = batch_size,
            callbacks = [checkpointer, reduce_lr],
            validation_data = (data_test, data_test)
        )
    
    if(False):
        metrics = model.autoencoder.evaluate(data_test, data_test, batch_size=512)
        print (model.autoencoder.metrics_names)
        print (metrics)
        

if __name__ == '__main__':
    gParameters=initialize_parameters()
    print("initialize_parameters set optimizer to ", gParameters['optimizer'])
    run(gParameters)



