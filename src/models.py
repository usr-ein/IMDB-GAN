import keras.backend as K
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.models import Model


def build_generative(noise_size, max_len, vocab_size):
    inputs = Input(shape=(noise_size,))
    x = inputs
    x = RepeatVector(max_len)(x)
    # x = LSTM(vocab_size, return_sequences=True)(x)
    predictions = LSTM(vocab_size, return_sequences=True)(x)

    generative_model = Model(inputs=inputs, outputs=predictions)
    return generative_model


def build_discriminative(max_len, vocab_size):
    inputs = Input(shape=(max_len, vocab_size,))
    x = inputs
    # DO NOT use embedding as it doesn't support onehot vectors and is just a lookup table in Keras
    # Instead, use a Dense layer that will project the words into their output space
    x = Dense(1000)(x)
    x = LSTM(256, return_sequences=False)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    discriminative_model = Model(inputs=inputs, outputs=predictions)
    return discriminative_model


def build_gan(noise_size, discriminative_model, generative_model):
    make_trainable(discriminative_model, False)
    inputs = Input(shape=(noise_size,))
    x = inputs
    x = generative_model(x)
    predictions = discriminative_model(x)
    gan = Model(inputs=inputs, outputs=predictions)
    return gan


def change_lr(model, val):
    K.set_value(model.optimizer.lr, val)


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
