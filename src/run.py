import os

from keras.models import load_model

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
    list_files.append('all of them')
    print("Please select weight file to load amongst:")
    for fn, i in zip(list_files, range(len(list_files))):
        print('({}) {}'.format(i, fn))
    index = input('Choice (type digit): ')
    while not index.isdigit() or int(index) < 0 or int(index) >= len(list_files):
        print('Please enter a valid value')
        index = input('Choice (type digit): ')
    index = int(index)

    return 'data/' + list_files[index]


def ask_input_file() -> str:
    pass


def load_inputs(fn: str):
    pass


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
    run(model, load_inputs(ask_input_file()))


def run_discriminative(model, inputs):
    pass


def run_generative(model, inputs):
    pass


if __name__ == '__main__':
    main()
