import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
import numpy as np

class lstmAutoencoder:
    """
    lstmAutoencoder represents a deep LSTM autoencoder architecture
    with mirrored encoder and decoder components. The central bottleneck is
    created by collapsing the incoming sequence to a single element.
    """
    def __init__(self,
                 input_shape,
                 num_units):
        self.input_shape = input_shape
        self.num_units = num_units
        self.encoder = None
        self.decoder = None
        self.model = None
        self.num_lstm_layers = len(num_units)
        self._model_input = None
        print('This is my LSTM autoencoder.')
        self.build_network()


    def summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())
        print(self.model.summary())


    def build_network(self):
        """
        Build the complete LSTM autoencoder architecture.
        """
        self.build_encoder()
        self.build_decoder()
        self.build_model()


    def build_model(self):
        """
        Build the complete LSTM autoencoder model.
        """
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='lstm_autoencoder')

    ######################## ENCODER ###########################

    def build_encoder(self):
        """
        Build complete encoder architecture.
        """
        enc_input = self.build_encoder_input()
        enc_lstms = self.build_encoder_lstms(enc_input)
        enc_latent = self.build_encoder_latent(enc_lstms)
        self._model_input = enc_input
        self.encoder = Model(enc_input, enc_latent, name='encoder')


    def build_encoder_input(self):
        """
        Build encoder input layer.
        """
        return layers.Input(shape=self.input_shape, name="encoder_input")


    def build_encoder_lstms(self, enc_input):
        """
        Create all encoder LSTM blocks before bottleneck.
        """
        x = enc_input
        for layer_index in range(self.num_lstm_layers - 1):
            x = self.build_encoder_lstm(layer_index, x)
        return x


    def build_encoder_lstm(self, layer_index, x):
        """
        Create single LSTM block that returns sequences.
        """
        lstm_layer = layers.LSTM(
            units=self.num_units[layer_index],
            activation='tanh',
            return_sequences=True,
            name=f'enc_lstm_layer_{layer_index+1}'
        )
        x = lstm_layer(x)
        return x


    def build_encoder_latent(self, x):
        """
        Create final encoder LSTM block that does not return sequences.
        """
        lstm_layer = layers.LSTM(
            units=self.num_units[-1],
            activation='tanh',
            return_sequences=False,
            name=f'enc_output'
        )
        x = lstm_layer(x)
        return x

    ######################## DECODER ###########################

    def build_decoder(self):
        """
        Build complete decoder architecture.
        """
        dec_input = self.build_decoder_input()
        dec_repeat = self.build_repeat_layer(dec_input)
        dec_lstms = self.build_decoder_lstms(dec_repeat)
        dec_output = self.build_decoder_output(dec_lstms)
        self.decoder = Model(dec_input, dec_output, name='decoder')


    def build_decoder_input(self):
        """
        Build decoder input layer.
        """
        return layers.Input(shape=self.num_units[-1], name='decoder_input')


    def build_repeat_layer(self, dec_input):
        """
        Repeat decoder input to generate sequence.
        """
        return layers.RepeatVector(self.input_shape[0], name='repeat_layer')(dec_input)


    def build_decoder_lstms(self, dec_repeat):
        """
        Create all decoder LSTM blocks.
        """
        x = dec_repeat
        for layer_index in reversed(range(self.num_lstm_layers - 1)):
            x = self.build_decoder_lstm(layer_index, x)
        return x


    def build_decoder_lstm(self, layer_index, x):
        """
        Create single decoder LSTM block.
        """
        lstm_layer = layers.LSTM(
            units=self.num_units[layer_index],
            activation='tanh',
            return_sequences=True,
            name=f'dec_lstm_layer_{self.num_lstm_layers - layer_index}'
        )
        x = lstm_layer(x)
        return x


    def build_decoder_output(self, dec_lstms):
        """
        Create decoder output from time distributed dense layers.
        """
        dense_layer = layers.TimeDistributed(layers.Dense(units=self.input_shape[1], activation='linear'),
                                             name='decoder_output')
        return dense_layer(dec_lstms)