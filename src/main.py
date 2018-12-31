import json

import keras.backend as K
import numpy as np
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import sequence


def main():
    np.random.seed(0)

    # Quick run
    # vocab_size = 1000
    # max_len = 600
    # n_samples = 128

    vocab_size = 2000
    max_len = 1000
    n_samples = 128

    noise_size = 100
    data, labels = load_data(max_len=max_len, vocab_size=vocab_size, n_samples=n_samples)

    generative_optimizer = Adam(lr=1e-4)
    discriminative_optimizer = Adam(lr=1e-3)

    # In: (None, 100) <-- rnd noise
    # Out: (None, 1000, 2000) <-- sentence of (max_len) words encoded in (vocab_size)
    generative_model = build_generative(noise_size, max_len, vocab_size)
    generative_model.compile(loss='binary_crossentropy', optimizer=generative_optimizer)
    # print(generative_model.summary())

    # In: (None, 1000, 2000) <-- sentence of (max_len) words encoded in (vocab_size)
    # Out: (None, 1) <-- probability of the sentence being real
    discriminative_model = build_discriminative(max_len, vocab_size)
    discriminative_model.compile(loss='binary_crossentropy', optimizer=discriminative_optimizer)
    # print(discriminative_model.summary())

    # Stacked GAN
    # In: (None, 100) <-- rnd noise
    # Out: (None, 1) <-- probability of the sentence being real
    make_trainable(discriminative_model, False)
    inputs = Input(shape=(noise_size,))
    x = inputs
    x = generative_model(x)
    predictions = discriminative_model(x)
    gan = Model(inputs=inputs, outputs=predictions)
    gan.compile(loss='binary_crossentropy', optimizer=generative_optimizer)
    # print(gan.summary())

    # -- Training the discriminator alone
    print('=' * 10 + 'Training discriminative model' + '=' * 10)

    print('-' * 10 + 'Building training data' + '-' * 10)
    training_samples, training_outputs = generate_mixed_data(data.train,
                                                             generative_model,
                                                             noise_size=noise_size,
                                                             vocab_size=vocab_size,
                                                             real_samples_size=100,
                                                             generated_samples_size=100)
    print('Training samples shape: ', training_samples.shape)
    print('Training outputs shape: ', training_outputs.shape)

    print('-' * 10 + 'Building testing data' + '-' * 10)
    testing_samples, testing_outputs = generate_mixed_data(data.test,
                                                           generative_model,
                                                           noise_size=noise_size,
                                                           vocab_size=vocab_size,
                                                           real_samples_size=100,
                                                           generated_samples_size=100)
    print('Testing samples shape: ', testing_samples.shape)
    print('Testing outputs shape: ', testing_outputs.shape)

    print('-' * 10 + 'Running the training process' + '-' * 10)
    make_trainable(discriminative_model, True)
    discriminative_model.fit(training_samples, training_outputs, batch_size=128, epochs=2)

    print('-' * 10 + 'Evaluating the discriminative model' + '-' * 10)
    scores = discriminative_model.evaluate(testing_samples, testing_outputs)
    print('Loss on testing samples: {:.2%}'.format(scores))

    losses = {"d": [], "g": []}

    try:
        change_lr(gan, 1e-4)
        change_lr(discriminative_model, 1e-3)

        losses = train(nb_epochs=6000, batch_size=32,
                       training_data=data.train,
                       discriminative_model=discriminative_model,
                       generative_model=generative_model,
                       gan_model=gan,
                       noise_size=noise_size,
                       vocab_size=vocab_size,
                       losses=losses)
        export('1', losses, discriminative_model, generative_model, gan)

        change_lr(gan, 1e-5)
        change_lr(discriminative_model, 1e-4)

        losses = train(nb_epochs=2000, batch_size=32,
                       training_data=data.train,
                       discriminative_model=discriminative_model,
                       generative_model=generative_model,
                       gan_model=gan,
                       noise_size=noise_size,
                       vocab_size=vocab_size,
                       losses=losses)
        export('2', losses, discriminative_model, generative_model, gan)

        change_lr(gan, 1e-6)
        change_lr(discriminative_model, 1e-5)

        losses = train(nb_epochs=2000, batch_size=32,
                       training_data=data.train,
                       discriminative_model=discriminative_model,
                       generative_model=generative_model,
                       gan_model=gan,
                       noise_size=noise_size,
                       vocab_size=vocab_size,
                       losses=losses)
        export('3', losses, discriminative_model, generative_model, gan)

    except KeyboardInterrupt as _:
        export('quitedInBetween', losses, discriminative_model, generative_model, gan)


def export(phase, losses, discriminative_model, generative_model, gan):
    export_losses(losses, fn="metrics_phase{}.json".format(phase))

    print('Saving the discriminative model')
    discriminative_model.save('discriminative_model_phase{}.h5'.format(phase))
    print('Saving the generative model')
    generative_model.save('generative_model_phase{}.h5'.format(phase))
    print('Saving the gan model')
    gan.save('gan_phase{}.h5'.format(phase))


def export_losses(losses, fn="metrics.json", verbose=True):
    if verbose:
        print('Exporting metrics')
    # Saves the losses for latter plotting
    with open(fn, "w") as f:
        json.dump(losses, f)


def train(training_data, nb_epochs, batch_size, discriminative_model, generative_model, gan_model, noise_size,
          vocab_size, losses):
    for epoch in range(nb_epochs):
        d_training_samples, d_training_outputs = generate_mixed_data(training_data,
                                                                     generative_model,
                                                                     noise_size=noise_size,
                                                                     vocab_size=vocab_size,
                                                                     real_samples_size=batch_size,
                                                                     generated_samples_size=batch_size)
        make_trainable(discriminative_model, True)
        d_loss = discriminative_model.train_on_batch(
            x=d_training_samples,
            y=d_training_outputs)

        losses['d'].append(float(d_loss))

        gan_training_samples = np.random.uniform(0, 1, size=[batch_size, noise_size])
        gan_training_outputs = np.zeros([batch_size, 1])

        make_trainable(discriminative_model, False)
        gan_loss = gan_model.train_on_batch(
            x=gan_training_samples,
            y=gan_training_outputs)

        losses['g'].append(float(gan_loss))

        print('Epoch {}/{}\td_loss {:.3%}\tgan_loss {:.3%}'.format(epoch, nb_epochs, d_loss, gan_loss))
        export_losses(losses, verbose=False)

    return losses


def change_lr(model, val):
    K.set_value(model.optimizer.lr, val)


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


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


def to_one_hot(x, num_classes=None, dtype='int'):
    # x.shape = (n_samples, max_len) and x[0, :] = [3, 23, 12, 45, .., n] with n <= max_int
    input_shape = x.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    x = x.ravel()
    if not num_classes:
        num_classes = np.max(x) + 1
    n = x.shape[0]
    one_hot = np.zeros((n, num_classes), dtype=dtype)
    one_hot[np.arange(n), x] = 1
    output_shape = input_shape + (num_classes,)
    one_hot = np.reshape(one_hot, output_shape)
    return one_hot


def generate_mixed_data(real_data, generative_model, noise_size, vocab_size, real_samples_size=100,
                        generated_samples_size=100):
    # Real samples
    samples_idx = np.random.randint(real_data.shape[0], size=real_samples_size)
    real_samples = to_one_hot(real_data[samples_idx, :], num_classes=vocab_size)
    # print('Real samples shape: ', real_samples.shape)

    # Fake samples
    # print('Generating {} fake samples from noise...'.format(generated_samples_size))
    noise_gen = np.random.uniform(0, 1, size=[generated_samples_size, noise_size])
    generated_samples = generative_model.predict(noise_gen)
    # print('Generated samples shape: ', generated_samples.shape)

    # Total samples
    input_samples = np.concatenate((real_samples, generated_samples))
    outputs = np.concatenate((np.ones(real_samples_size), np.ones(generated_samples_size)))
    # print('Samples shape: ', input_samples.shape)
    # print('Outputs shape: ', outputs.shape)

    return input_samples, outputs


def load_data(vocab_size=2000, max_len=1000, n_samples=128):
    input_data = lambda: None
    output_labels = lambda: None
    (input_data.train, output_labels.train), (input_data.test, output_labels.test) = imdb.load_data(
        num_words=vocab_size, index_from=0)

    input_data.train = sequence.pad_sequences(input_data.train, maxlen=max_len)
    input_data.test = sequence.pad_sequences(input_data.test, maxlen=max_len)

    assert n_samples <= 25000  # max: 25000 for IMDB

    input_data.train = input_data.train[:n_samples, ]
    output_labels.train = output_labels.train[:n_samples, ]

    input_data.test = input_data.test[:n_samples, ]
    output_labels.test = output_labels.test[:n_samples, ]

    return input_data, output_labels


if __name__ == '__main__':
    main()
