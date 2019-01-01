import h5py
import numpy as np
from keras.datasets import imdb


def ask_preds_file() -> str:
    list_files = [fn for fn in os.listdir('data') if ('.h5' in fn or '.h5py' in fn or '.hdf5' in fn)]
    print("Please select input file to load amongst:")
    for fn, i in zip(list_files, range(len(list_files))):
        print('({}) {}'.format(i, fn))
    index = input('Choice (type digit): ')
    while not index.isdigit() or int(index) < 0 or int(index) >= len(list_files):
        print('Please enter a valid value')
        index = input('Choice (type digit): ')
    index = int(index)

    return 'data/preds/' + list_files[index]


def load_preds(fn: str) -> np.ndarray:
    h5f = h5py.File(fn, 'r')
    if 'preds' not in h5f:
        print('\'preds\' HDF5 dataset not found in {}, exiting now...'.format(fn))
        exit()
    preds = h5f['preds'][:]
    h5f.close()
    preds = np.array(preds)
    return preds


def get_idx_to_word():
    word_to_idx = imdb.get_word_index()
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    idx_to_word[0] = ''
    idx_to_word[1] = ''
    idx_to_word[2] = ''
    return idx_to_word


def convert_generative_preds_to_int_vect(preds):
    decoded = np.zeros((preds.shape[0], preds.shape[1]))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            decoded[i, j] = np.argmax(preds[i, j])
    return decoded


def decode_preds(fn):
    preds = load_preds(fn)
    preds = convert_generative_preds_to_int_vect(preds)
    idx_to_word = get_idx_to_word()
    output_fn = fn[:fn.rfind('.')] + '.txt'
    print('output_fn', output_fn)
    with open(output_fn, 'a') as output_f:
        for row in preds:
            output_f.write(' '.join([idx_to_word[i] for i in row]) + '\n')
    print('Predictions written to ', output_fn)
    print('First review :')
    print(' '.join([idx_to_word[i] for i in preds[0]]))


if __name__ == '__main__':
    decode_preds(ask_preds_file())
