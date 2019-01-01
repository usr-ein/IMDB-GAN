import json

import h5py
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence


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


def export(phase, losses, discriminative_model, generative_model, gan):
    export_losses(losses, fn="data/metrics/metrics_phase{}.json".format(phase))

    print('Saving the discriminative model')
    discriminative_model.save('data/models/discriminative_model_phase{}.h5'.format(phase))
    print('Saving the generative model')
    generative_model.save('data/models/generative_model_phase{}.h5'.format(phase))
    print('Saving the gan model')
    gan.save('data/models/gan_phase{}.h5'.format(phase))


def export_losses(losses, fn="data/metrics/metrics.json", verbose=True):
    if verbose:
        print('Exporting metrics')
    # Saves the losses for latter plotting
    with open(fn, "w") as f:
        json.dump(losses, f)


def export_preds(preds, fn="data/preds/preds.h5"):
    h5f = h5py.File(fn, 'w')
    h5f.create_dataset('preds', data=preds)
    h5f.close()
