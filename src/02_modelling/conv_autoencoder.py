import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
import numpy as np


class ConvAutoencoder:
    """
    A Deep Convolutional Autoencoder
    """

    def __init__(self,
                 input_shape,
                 num_nodes,
                 num_kernel,
                 num_strides,
                 latent_dim):

        self.input_shape = input_shape
        self.num_nodes = num_nodes
        self.num_kernel = num_kernel
        self.num_strides = num_strides
        self.latent_dim = latent_dim

        self.num_layers = len(self.num_nodes)
        self._model_input = None
        self.shape_before_flatten = None

        self.encoder = None
        self.decoder = None
        self.model = None
        print("This is my autoencoder")

        self.build_network()

    def summary(self):
        self.encoder.summary()
        print("Shape before flatten={}".format(self.shape_before_flatten))
        self.decoder.summary()
        self.model.summary()

    def build_network(self):
        """
        Build the complete Autoencoder
        """
        self.build_encoder()
        self.build_decoder()
        self.build_model()

    ######################## ENCODER ###########################

    def build_encoder(self):
        """
        Build the encoder architecture
        """
        enc_input = self.build_encoder_input()

        enc_conv = self.build_encoder_convs(enc_input)

        enc_latent = self.build_encoder_latent(enc_conv)

        self._model_input = enc_input

        self.encoder = Model(enc_input, enc_latent,
                             name="Encoder")

    def build_encoder_input(self):
        """
        Build the input layer of Encoder
        """

        return layers.Input(shape=self.input_shape,
                            name="Encoder_input")

    def build_encoder_convs(self, enc_inp_layer):
        """
        Build the convolutional layers.
        As many layers as the 'num_layers'.
        """
        stack_layer = enc_inp_layer

        for layer_index in range(self.num_layers):
            stack_layer = self.build_encoder_conv(layer_index,
                                                  stack_layer)
        return stack_layer

    def build_encoder_conv(self, layer_ind, layer):
        """
        Build one particular Convolutional layer,
        Activation Function and Regularization.
        """
        # Convolutional layers
        X = layers.Conv2D(filters=self.num_nodes[layer_ind],
                          kernel_size=self.num_kernel[layer_ind],
                          strides=self.num_strides[layer_ind],
                          padding="same",
                          name="Conv_enc{}".format(layer_ind))(layer)

        # Batch Normalization
        X = layers.BatchNormalization(name="BatchNorm_enc{}"
                                      .format(layer_ind))(X)

        # ReLU activation function
        X = layers.ReLU(name="ReLU_enc{}".format(layer_ind))(X)

        return X

    def build_encoder_latent(self, conv_output):
        """
        Build the latent dimensional space.
        Embeds the features from the Conv2D layers
        into a lower dimensional space.
        """
        self.shape_before_flatten = K.int_shape(conv_output)[1:]

        """X = layers.Conv2D(filters=self.latent_dim,
                          kernel_size=(5,5),
                          strides=(6, 6),
                          padding='same'
                          )(conv_output)"""

        #X = layers.Flatten(name="Flatten_enc")(X)
        X = layers.Flatten(name="Flatten_enc")(conv_output)

        X = layers.Dense(self.latent_dim, name="Latent_enc")(X)

        return X
    ################################################################

    ######################## DECODER ###############################

    def build_decoder(self):
        """
        Build the decoder architecture
        """

        dec_input = self.build_decoder_input()
        dec_dense = self.build_decoder_dense(dec_input)
        dec_reshape = self.build_decoder_reshape(dec_dense)
        dec_conv = self.build_decoder_convs(dec_reshape)
        dec_output = self.build_decoder_output(dec_conv)

        self.decoder = Model(dec_input, dec_output,
                             name='Decoder')

    def build_decoder_input(self):
        """
        Input layer of decoder of shape latent_dim
        """
        decoder_input_layer = layers.Input(
            shape=(self.latent_dim,),
            name="decoder_input")

        return decoder_input_layer

    def build_decoder_dense(self, decoder_input):
        """
        Dense layer after embedding space
        Shape should be the 'combined' array
        of Encoder before flattening i.e.
        np.prod(self.shape_before_flatten)
        """
        decoder_dense_shape = np.prod(self.shape_before_flatten)
        """decoder_dense_shape = self.shape_before_flatten[1] * \
                              self.shape_before_flatten[2] * \
                              self.shape_before_flatten[3]"""

        decoder_dense = layers.Dense(decoder_dense_shape,
                                     name="Decoder_dense")(decoder_input)
        return decoder_dense

    def build_decoder_reshape(self, decoder_dense):
        """
        Reshape Layer after Flattening to feed in
        to the Conv2DTranspose layer.
        """
        reshaped = layers.Reshape(self.shape_before_flatten)
        return reshaped(decoder_dense)

    def build_decoder_convs(self, decoder_reshape):
        """
        Build all the Conv2DTranspose layers
        of shape equal to number of layers
        excluding the first.
        """
        stack_layer_dec = decoder_reshape

        for layer_index in reversed(range(0, self.num_layers-1)):
            # for layer_index in range(0, self.num_layers-1):
            stack_layer_dec = self.build_decoder_conv(layer_index,
                                                      stack_layer_dec)
        return stack_layer_dec

    def build_decoder_conv(self, layer_ind, X):
        """
        Build individual Conv2DTranspose layer of
        the Decoder.
        """
        dec_convlayer_num = self.num_layers - layer_ind

        conv_trans = layers.Conv2DTranspose(filters=self.num_nodes[layer_ind],
                                            kernel_size=self.num_kernel[layer_ind],
                                            strides=self.num_strides[layer_ind+1],
                                            padding="same",
                                            name="Decoder_ConvTrans{}"
                                            .format(dec_convlayer_num)
                                            )
        X = conv_trans(X)

        X = layers.BatchNormalization(name="Decoder_BatchNorm{}"
                                      .format(dec_convlayer_num))(X)

        X = layers.ReLU(name="Decoder_ReLU{}"
                        .format(dec_convlayer_num))(X)

        return X

    def build_decoder_output(self, X):
        """
        Output layer of the decoder
        """
        decoder_outlayer = layers.Conv2DTranspose(filters=1,
                                                  kernel_size=self.num_kernel[0],
                                                  strides=self.num_strides[0],
                                                  padding="same",
                                                  name="Decoder_output")
        X = decoder_outlayer(X)
        #decoder_final = layers.Activation("sigmoid", name="Decoder_activation")
        return X  # decoder_final(X)

    ######################## AUTOENCODER #######################################
    def build_model(self):
        """
        Build the Autoencoder model
        combining encoder and decoder
        """
        model_input = self._model_input
        encoder_output = self.encoder(model_input)
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)


if __name__ == '__main__':
    print("Importing Convolutional Autoencoder...")
