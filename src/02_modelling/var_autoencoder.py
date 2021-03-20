import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
import numpy as np

from ngdlm import models as ngdlmodels
from ngdlm import utils as ngdlutils
tf.compat.v1.disable_eager_execution()


class VarAutoencoder:
    """
    Variational Autoencoder
    """

    def __init__(self,
                 input_shape,
                 latent_dim,
                 num_nodes,
                 num_kernel,
                 num_strides):

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.num_kernel = num_kernel
        self.num_strides = num_strides

        self.num_layers = len(self.num_nodes)

        self.shape_before_flatten = None
        self._encoder_input = None

        self.encoder = None
        self.decoder = None
        self.model = None

        self.model_network()

    def summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())
        print(self.model.summary())

    def model_network(self):
        self.build_encoder()
        self.build_decoder()
        self.build_model()

    def build_model(self):
        """
        Build the VAE gluing the
        encoder and decoder using
        the ngdlmodels.
        """
        encoder = self.encoder
        decoder = self.decoder
        latent_dim = self.latent_dim

        self.model = ngdlmodels.VAE(
            encoder,
            decoder,
            latent_dim=latent_dim
        )

    def build_decoder(self):
        """
        Build the decoder architecture
        """
        dec_input = self.build_decoder_input()
        dec_dense = self.build_decoder_dense(dec_input)
        dec_reshape = self.build_decoder_reshape(dec_dense)
        dec_deconv = self.build_decoder_deconvs(dec_reshape)
        dec_output = self.build_decoder_output(dec_deconv)

        self.decoder = Model(dec_input,
                             dec_output,
                             name='decoder')

    def build_decoder_input(self):
        """
        Return decoder input layer
        i.e. the latent dim layer
        """

        return layers.Input(shape=(self.latent_dim),
                            name='decoder_input')

    def build_decoder_dense(self, dec_input):
        """
        Decoder dense with as many nodes
        as the shape of the convolution
        before flattening.
        """
        dense_shape = np.prod(self.shape_before_flatten)

        dense = layers.Dense(dense_shape,
                             name='dense_dec')

        return dense(dec_input)

    def build_decoder_reshape(self, dec_dense):
        """
        Reshape Decoder dense layer to apply
        deconvolution.
        """
        shape = self.shape_before_flatten

        return layers.Reshape(shape, name='reshape_dec')(dec_dense)

    def build_decoder_deconvs(self, dec_reshape):
        """
        Loop for building the deconvolutional layers
        """
        stack_layer = dec_reshape

        for layer_index in reversed(range(self.num_layers - 1)):
            stack_layer = self.build_decoder_deconv(layer_index,
                                                    stack_layer)

        return stack_layer

    def build_decoder_deconv(self, layer_index, layer):
        """
        One block of deconvolutional layer
        """
        conv = layers.Conv2DTranspose(filters=self.num_nodes[layer_index],
                                      kernel_size=self.num_kernel[layer_index],
                                      strides=self.num_strides[layer_index + 1],
                                      padding='same',
                                      name='convTranspose_dec{}'.format(layer_index))

        X = conv(layer)
        X = layers.BatchNormalization(name='batchnorm_dec{}'.format(layer_index))(X)
        X = layers.ReLU(name='relu_dec{}'.format(layer_index))(X)

        return X

    def build_decoder_output(self, dec_deconv):
        """
        Decoder output layer.
        Reconstructed image.
        """
        dec_out = layers.Conv2DTranspose(filters=1,
                                         kernel_size=self.num_kernel[0],
                                         strides=self.num_strides[0],
                                         padding='same',
                                         name='output_dec')
        return dec_out(dec_deconv)

    def build_encoder(self):
        """
        Build encoder architecture
        """
        enc_input = self.build_encoder_input()
        enc_conv = self.build_encoder_convs(enc_input)
        enc_latent = self.build_encoder_latent(enc_conv)

        # self.shape_before_flatten = K.int_shape(enc_output)[1:]

        self._encoder_input = enc_input

        self.encoder = Model(enc_input,
                             enc_latent,
                             name='encoder')

    def build_encoder_input(self):
        """
        Input layer of encoder.
        The input image is fed here.
        """
        return layers.Input(shape=self.input_shape,
                            name='encoder_input')

    def build_encoder_convs(self, conv_enc_input):
        """
        Loop for building the convolutional
        layers of the encoder
        """
        stack_layer = conv_enc_input

        for layer_index in range(self.num_layers):
            stack_layer = self.build_encoder_conv(layer_index,
                                                  stack_layer)

        return stack_layer

    def build_encoder_conv(self, layer_index, layer):
        """
        One convolutional layer block
        """
        conv = layers.Conv2D(filters=self.num_nodes[layer_index],
                             kernel_size=self.num_kernel[layer_index],
                             strides=self.num_strides[layer_index],
                             padding='same',
                             name='conv_enc{}'.format(layer_index))

        X = conv(layer)
        X = layers.BatchNormalization(name='batchnorm_enc{}'.format(layer_index))(X)
        X = layers.ReLU(name='relu_enc{}'.format(layer_index))(X)

        return X

    def build_encoder_latent(self, enc_conv):
        """
        Flatten the convolutional layers
        and encode the latent space.
        """
        self.shape_before_flatten = K.int_shape(enc_conv)[1:]

        X = layers.Flatten(name='flatten_enc')(enc_conv)

        return layers.Dense(self.latent_dim, name='latent_enc')(X)
