import os
import h5py
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Dropout, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback

tf.compat.v1.disable_eager_execution()

with h5py.File("/local_storage/user/data/norm_dataset_itop.hdf5", 'r') as hf:
    x_train = np.asarray(hf['x_train'])
    y_train = np.asarray(hf['y_train'])
    x_test  = np.asarray(hf['x_test'])
    y_test  = np.asarray(hf['y_test'])
    
train_size, img_w, img_h = x_train.shape
test_size, j_type, j_coord = y_test.shape

# (num_samples, 192, 112, 1)
x_train = x_train.reshape(-1, img_w, img_h, 1)
x_test  = x_test.reshape(-1,  img_w, img_h, 1)

# (num_samples, 45, 1)
y_train = y_train.reshape(-1, j_type*j_coord, 1)
y_test  = y_test.reshape(-1,  j_type*j_coord, 1)


"""
# Beta Annealing Variational Autoencoder

"""
beta = K.variable(value=0.0)

def calculate_reconstruction_loss(y_target, y_predicted):
    error = y_target - y_predicted
    reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
    return reconstruction_loss

def calculate_kl_loss(model):
    def _calculate_kl_loss(*args):
        kl_loss = -0.5 * K.sum(1 + model.log_variance - K.square(model.mu) -
                               K.exp(model.log_variance), axis=1)
        return beta * kl_loss
    return _calculate_kl_loss

class BetaAnnealingScheduler:
    def __init__(self, n_epoch, anneal_type):
        self.n_epoch = n_epoch

        self.start = 0.0
        self.stop = 1.0
        self.c = 1.0

        if anneal_type == "sigmoid":
            self.L = self.frange_cycle_sigmoid()
        elif anneal_type == "cosine":
            self.L = self.frange_cycle_cosine()
        else: # Linear
            self.L = self.frange_cycle_linear()

    def frange_cycle_linear(self, n_cycle=4, ratio=0.5):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / n_cycle
        step = (self.stop - self.start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            v, i = self.start, 0
            while v <= self.stop and (int(i + c * period) < self.n_epoch):
                L[int(i + c * period)] = v
                v += step
                i += 1

        self.L = L * self.c
        return self.L

    def frange_cycle_sigmoid(self, n_cycle=4, ratio=0.5):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / n_cycle
        step = (self.stop - self.start) / (period * ratio)  # step is in [0,1]

        for c in range(n_cycle):
            v, i = self.start, 0
            while v <= self.stop:
                L[int(i + c * period)] = 1.0 / (1.0 + np.exp(-(v * 12.0 - 6.0)))
                v += step
                i += 1

        self.L = L * self.c
        return self.L

    def frange_cycle_cosine(self, n_cycle=4, ratio=0.5):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / n_cycle
        step = (self.stop - self.start) / (period * ratio)  # step is in [0,1]

        for c in range(n_cycle):

            v, i = self.start, 0
            while v <= self.stop:
                L[int(i + c * period)] = 0.5 - 0.5 * np.cos(v * np.pi)
                v += step
                i += 1

        self.L = L * self.c
        return self.L
    
    def __call__(self, i, *args, **kwargs):
        return self.L[i]

class VAE:
  def __init__(self,
               input_image, #shape of the input data
               conv_filters, #convolutional network filters
               conv_kernels, #convNet kernel size
               conv_strides, #convNet strides
               latent_space_dim):
    self.input_image = input_image # (192, 112)
    self.conv_filters = conv_filters # is a list for each layer, i.e. [2, 4, 8]
    self.conv_kernels = conv_kernels # list of kernels per layer, [1, 2, 3]
    self.conv_strides = conv_strides # stride for each filter [1, 2, 2], note: 2 means you are downsampling the data in half
    self.latent_space_dim = latent_space_dim # how many neurons on bottleneck
    self.reconstruction_loss_weight = 100000

    self.encoder = None
    self.decoder = None
    self.model = None
    self.hist = None

    self._num_conv_layers = len(conv_filters)
    self._shape_before_bottleneck = None
    self._model_output = None
    self._model_input = None
    self._foi_input = None

    self._build()

  def summary(self):
    self.encoder.summary()
    print("\n")
    self.decoder.summary()
    print("\n")
    self.model.summary()

  def _build(self):
    self._build_encoder()
    self._build_decoder()
    self._build_autoencoder()

  def compile(self, learning_rate=0.0001):
    optimizer = Adam(learning_rate=learning_rate)
    self.model.compile(optimizer=optimizer, loss=self._calculate_combined_loss,
                       metrics=[calculate_reconstruction_loss, calculate_kl_loss(self)])
  
  
  def train(self, x_train, batch_size, num_epochs):
    self.hist = self.model.fit(x_train, x_train,
                               batch_size=batch_size,
                               epochs=num_epochs,
                               shuffle=True)
    
  def _calculate_combined_loss(self, y_target, y_predicted):
    reconstruction_loss = calculate_reconstruction_loss(y_target, y_predicted)
    kl_loss = calculate_kl_loss(self)()
    combined_loss = reconstruction_loss * self.reconstruction_loss_weight + kl_loss
    return combined_loss

  #----------------FULL MODEL-----------------#
  def _build_autoencoder(self):
    model_input = self._model_input
    model_output = self.decoder(self.encoder(model_input))
    self.model = Model(model_input, model_output, name='autoencoder')

  #----------------DECODER-----------------#
  def _build_decoder(self):
    decoder_input = self._add_decoder_input()
    dense_layer = self._add_dense_layer(decoder_input)
    reshape_layer = self._add_reshape_layer(dense_layer)
    conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
    decoder_output = self._add_decoder_output(conv_transpose_layers)
    self.decoder = Model(decoder_input, decoder_output, name="decoder")
    self._model_output = self.decoder(self.encoder(self._model_input))
    

  def _add_decoder_input(self):
    return Input(shape=self.latent_space_dim, name="decoder_input")

  def _add_dense_layer(self, decoder_input):
    num_neurons = np.prod(self._shape_before_bottleneck) # [ 1, 2, 4] -> 8
    dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
    return dense_layer

  def _add_reshape_layer(self, dense_layer):
    return Reshape(self._shape_before_bottleneck)(dense_layer)

  def _add_conv_transpose_layers(self, x):
    for layer_index in reversed(range(1, self._num_conv_layers)):
      x = self._add_conv_transpose_layer(layer_index, x)
    return x

  def _add_conv_transpose_layer(self, layer_index, x):
    layer_num = self._num_conv_layers - layer_index
    conv_transpose_layer = Conv2DTranspose(
        filters=self.conv_filters[layer_index],
        kernel_size = self.conv_kernels[layer_index],
        strides = self.conv_strides[layer_index],
        activation='relu',
        padding = "same",
        name=f"decoder_conv_transpose_layer_{layer_num}"
    )
    x = conv_transpose_layer(x)
    x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
    return x

  def _add_decoder_output(self, x):
    conv_transpose_layer = Conv2DTranspose(
        filters = 1,
        kernel_size = self.conv_kernels[0],
        strides = self.conv_strides[0],
        padding = "same",
        name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
    )
    x = conv_transpose_layer(x)
    output_layer = Activation("sigmoid", name="sigmoid_output_layer")(x)
    return output_layer

  #----------------ENCODER-----------------#
  def _build_encoder(self):
    encoder_input = self.input_image
    conv_layers = self._add_conv_layers(encoder_input)
    bottleneck =  self._add_bottleneck(conv_layers)
    self._model_input = encoder_input
    self.encoder = Model(encoder_input, bottleneck, name="encoder")

  def _add_conv_layers(self, encoder_input):
    """Creates all convolutional blocks in encoder"""
    x = encoder_input
    for layer_index in range(self._num_conv_layers):
      x = self._add_conv_layer(layer_index, x)
    return x
  
  def _add_conv_layer(self, layer_index, x):
    """
    Adds a convolutional block to a graph of layers, consisting
    of Conv 2d + ReLu activation + batch normalization.
    """
    layer_number = layer_index + 1
    conv_layer = Conv2D(
        filters= self.conv_filters[layer_index],
        kernel_size = self.conv_kernels[layer_index],
        strides = self.conv_strides[layer_index],
        activation='relu',
        padding = "same",
        name = f"encoder_conv_layer_{layer_number}"
    )
    x = conv_layer(x)
    x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
    return x

  #-------------LATTENT SPACE-------------#
  def _add_bottleneck(self, x):
    """Flatten data and add bottleneck with Gaussian sampling (Dense layer)"""
    self._shape_before_bottleneck = K.int_shape(x)[1:]
    x = Flatten()(x)
    self.mu = Dense(self.latent_space_dim,name="mu")(x)
    self.log_variance = Dense(self.latent_space_dim,
                              name="log_variance")(x)
    
    def sample_point_from_normal_distribution(args):
      mu, log_variance = args
      epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., stddev=1.)
      sampled_point = mu + K.exp(log_variance / 2) * epsilon

      return sampled_point

    x = Lambda(sample_point_from_normal_distribution, 
              name="encoder_output")([self.mu, self.log_variance])
    return x

def beta_annealing(epoch):
    value = BetaAnnealingScheduler(n_epoch=num_epochs_to_train, anneal_type='sigmoid')(epoch)
    # print("Beta value: {}".format(value))
    K.set_value(beta, value)

epoch_callback = LambdaCallback(on_epoch_begin=lambda epoch, log: beta_annealing(epoch))

"""
Model Classifier

"""

class MLP():
  def __init__(self, encoder, input_dim, input_image, label_input, latent_space_dim):
    self.encoder = encoder
    self.input_dim = input_dim
    self.input_image = input_image
    self.label_input = label_input
    self.latent_space_dim = latent_space_dim

    self.encoder.trainable = False
    
    self.model = None
    self.mlp_output = None
    self.mlp_model  = self._build_mlp()

    self._build()

  def _build(self):
    self.mlp_output = self.mlp_model(self.encoder(self.input_image, training=False))
    self.model = Model(inputs=self.input_image, outputs=self.mlp_output, name='vae_classifier')

  def _calculate_label_loss(self, y_target, y_predicted):
    error = y_target - y_predicted
    reconstruction_loss = K.mean(K.square(error), axis=0)
    return reconstruction_loss

  def compile(self, learning_rate=0.0001):
    self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=self._calculate_label_loss)
  
  def train(self, x_train, y_train, batch_size, num_epochs):
    self.hist = self.model.fit(x_train, y_train,
                               batch_size=batch_size,
                               epochs=num_epochs,
                               shuffle=True)

  def _build_mlp(self):
      mlp_input = self._add_mlp_input()
      mlp_output = self._add_mlp_layers(mlp_input)
      return Model(mlp_input, mlp_output, name="MLP_model")

  def _add_mlp_input(self):
    return Input(shape=self.latent_space_dim, name="mlp_input")

  def _add_mlp_layers(self, o_layer):
    dimension = self.input_dim

    while dimension >= 256:
       d_layer = Dense(dimension, activation='relu', name=f"dense_{dimension}")(o_layer)
       o_layer = Dropout(0.5, name=f"dropout_{dimension}")(d_layer)
       dimension = dimension / 2

    d_layer = Dense(45, activation='softmax', name='dense_45')(o_layer)
    r_layer = Reshape((45, 1), name='mlp_output')(d_layer)
    return r_layer

"""
Latent-space exploration

"""
learning_rate = 0.001
num_epochs_to_train = 20
batch_size = 128

input_image = Input(shape=(192, 112, 1), name='image_input')
input_label = Input(shape=(45, 1), name='label_input')

class Filter:
   def __init__(self, conv, limit):
      self.conv = conv
      self.limit = limit

vector_dim = [256, 512, 1024, 2048, 4096]

conv_filters = [
   Filter((512, 256, 128, 64, 32), 1344),
   Filter((256, 128,  64, 32, 16), 672),
   Filter((512, 256, 128, 64), 10752),
   Filter((256, 128,  64, 32), 5376)
]

conv_kernels = (3, 3, 3, 3, 3)
conv_strides = (2, 2, 2, 2, (2,1))

for filter in conv_filters:
   for dim in vector_dim:
      if dim > filter.limit: break
      kernel = conv_kernels if len(filter.conv) == 5 else conv_kernels[1:]
      stride = conv_strides if len(filter.conv) == 5 else conv_strides[1:]
      print("# {} --> {}/{}".format(filter.conv, dim, filter.limit))
      
      # VAE training
      vae = VAE(input_image = input_image, conv_filters=filter.conv, conv_kernels=kernel, conv_strides=stride, latent_space_dim=dim)
      print("Parameter count for VAE: {}".format(vae.model.count_params()))

      vae.compile(learning_rate)
      hist = vae.model.fit(x_train, x_train, callbacks=[epoch_callback], batch_size=batch_size, epochs=num_epochs_to_train, shuffle=True)
      print("Model loss: {}".format(hist.history['calculate_reconstruction_loss']))

      # MLP training
      model = MLP(encoder=vae.encoder, input_dim=dim, input_image=input_image, label_input=input_label, latent_space_dim=dim)
      print("Parameter count for MLP: {}".format(model.mlp_model.count_params()))

      model.compile(learning_rate=learning_rate)
      hist = model.train(x_train, y_train, batch_size, num_epochs_to_train)
      print("Model loss: {}".format(hist.history['loss']))