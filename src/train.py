import numpy as np
from keras.optimizers import Adam

# Data inputs (loading, generating data)
# and outputs (export models' weights, loss)
from data_io import export, export_losses
from data_io import load_data, generate_mixed_data
# Model building functions
from models import build_discriminative, build_generative, build_gan
from models import change_lr, make_trainable


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
    gan = build_gan(noise_size, discriminative_model, generative_model)
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
        export_losses(losses, fn='data/metrics/metrics.json', verbose=False)

    return losses


if __name__ == '__main__':
    main()
