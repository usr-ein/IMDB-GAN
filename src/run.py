import os
from typing import Union

import h5py
import numpy as np
from keras.models import load_model

from data_io import export_preds

list_model_types = ['discriminative', 'generative']


def ask_model_type() -> str:
    global list_model_types
    print("Please select the type of model amongst:")
    for fn, i in zip(list_model_types, range(len(list_model_types))):
        print('({}) {}'.format(i, fn))
    index = input('Choice (type digit): ')
    while not index.isdigit() or int(index) < 0 or int(index) >= len(list_files):
        print('Please enter a valid value')
        index = input('Choice (type digit): ')
    index = int(index)

    return list_model_types[index]


def ask_filename() -> str:
    list_files = [fn for fn in os.listdir('data') if ('.h5' in fn or '.h5py' in fn or '.hdf5' in fn)]
    print("Please select weight file to load amongst:")
    for fn, i in zip(list_files, range(len(list_files))):
        print('({}) {}'.format(i, fn))
    index = input('Choice (type digit): ')
    while not index.isdigit() or int(index) < 0 or int(index) >= len(list_files):
        print('Please enter a valid value')
        index = input('Choice (type digit): ')
    index = int(index)

    return 'data/' + list_files[index]


def ask_input_file() -> Union[str, None]:
    list_files = [fn for fn in os.listdir('data') if ('.h5' in fn or '.h5py' in fn or '.hdf5' in fn)]
    list_files += ['none']
    print("Please select input file to load amongst:")
    for fn, i in zip(list_files, range(len(list_files))):
        print('({}) {}'.format(i, fn))
    index = input('Choice (type digit): ')
    while not index.isdigit() or int(index) < 0 or int(index) >= len(list_files):
        print('Please enter a valid value')
        index = input('Choice (type digit): ')
    index = int(index)

    if list_files[index] == list_files[-1]:
        return None

    return 'data/' + list_files[index]


def load_inputs(fn: str) -> np.ndarray:
    h5f = h5py.File(fn, 'r')
    if 'inputs' not in h5f:
        print('\'inputs\' HDF5 dataset not found in {}, exiting now...'.format(fn))
        exit()
    inputs = h5f['inputs'][:]
    h5f.close()
    inputs = np.array(inputs)
    return inputs


def main():
    global list_model_types
    model_type = ask_model_type()
    fn = ask_filename()
    model = None
    try:
        model = load_model(fn)
    except OSError as _:
        print('Bad file format, be sure the file isn\'t corrupted')

    if model is None:
        print('Model failed to load for mysterious reasons')
        exit()

    if model_type == list_model_types[0]:
        run = run_discriminative
    elif model_type == list_model_types[1]:
        run = run_generative
    else:
        run = lambda _, __: print('Failed to determined action to take for requested type of model')
    inputs_fn = ask_input_file()
    inputs = None
    if inputs_fn is not None:
        inputs = load_inputs(inputs_fn)
    preds = run(model, inputs)
    fn_preds = 'preds_{}_{}.h5'.format(model_type, inputs_fn[inputs_fn.rfind('/'):inputs_fn.rfind('.')])
    export_preds(preds, fn=fn_preds)
    print('Predictions have been exported to {}'.format(fn_preds))
    return


def run_discriminative(model, inputs: np.ndarray) -> np.ndarray:
    if inputs is None:
        print('Inputs of discriminative mode cannot be None')
        exit()
    model_input_shape = model.layers[0].input_shape
    # Check if input is of the right shape
    assert model_input_shape[1:] == inputs.shape[1:]
    print('Running discriminative model')
    preds = model.predict(inputs)
    return preds


def run_generative(model, inputs: np.ndarray = None) -> np.ndarray:
    model_input_shape = model.layers[0].input_shape
    if inputs is not None:
        # Check if input is of the right shape
        assert inputs.shape[0] == 100
        assert model_input_shape[1:] == inputs.shape[1:]
    else:
        inputs = np.random.uniform(0, 1, size=[100, model_input_shape[-1]])
    preds = model.predict(inputs)
    return preds


if __name__ == '__main__':
    main()
