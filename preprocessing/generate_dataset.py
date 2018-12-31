import os
import re
from typing import List, Set, Dict

import h5py
import numpy as np


def set_working_dir():
    project_root_name = 'IMDB_GAN'
    cwd = os.getcwd()
    if cwd[cwd.rfind('/'):] == '/' + project_root_name:
        return
    elif project_root_name not in cwd:
        print('wrong path, went too far up')
        return
    else:
        os.chdir('..')
        set_working_dir()


def add_padding(dataset):
    max_col = max([len(list(a)) for a in dataset])
    max_row = len(dataset)

    result = np.zeros((max_row, max_col))
    for i in range(max_row):
        result[i, :len(dataset[i])] = np.array(dataset[i])

    return result.astype(np.uint32)


def vocab_dict_to_list(voc):
    items = voc.items()
    keys = [e[0] for e in items]
    values = [e[1] for e in items]
    return keys, values


def list_to_vocab_dict(k, v):
    return {w: i for w, i in zip(k, v)}


def load_data(input_file: str = 'cache/dataset.hdf5', size: float = 1.0):
    set_working_dir()

    x = lambda: None  # Obj that will contain the inputs
    y = lambda: None  # Obj that will contain the outputs

    if not os.path.isfile(input_file):
        print('Generating dataset from sources')
        generate_data(input_file, size)

    print('Loading generated dataset from cache')
    with h5py.File(input_file, 'r') as hf:
        x.train = np.array(hf["train/x"])
        y.train = np.array(hf["train/y"])

        x.test = np.array(hf["test/x"])
        y.test = np.array(hf["test/y"])

        k = hf["vocab_dict/keys"]
        k = [w.decode('utf8') for w in k]
        v = hf["vocab_dict/values"]

        vocab_dict = list_to_vocab_dict(k, v)

    return x, y, vocab_dict


def generate_data(output_file: str = 'cache/dataset.h5py', size: float = 1.0):
    set_working_dir()

    dataset_path = 'data/dataset'
    train_path = dataset_path + '/train'
    test_path = dataset_path + '/test'

    regex = re.compile(r'[^A-Za-z\s]')

    x_train, y_train, vocab_set_train = parse_files(train_path, regex=regex, size=size)
    x_test, y_test, vocab_set_test = parse_files(test_path, regex=regex, size=size)

    vocab_set = vocab_set_test | vocab_set_train
    # Leave 0 for padding
    vocab_dict = {w: i for w, i in zip(vocab_set, range(1, len(vocab_set) + 1))}
    vocab_dict[''] = 0

    x_train = add_padding(encode_dataset(x_train, vocab_dict))
    x_test = add_padding(encode_dataset(x_test, vocab_dict))

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    export_dataset(x_train, x_test, y_train, y_test, vocab_dict, output_file)


def export_dataset(x_train, x_test, y_train, y_test, vocab_dict, output_file: str):
    with h5py.File(output_file, 'w') as hf:
        training = hf.create_group("train")
        training.create_dataset("x", data=x_train)
        training.create_dataset("y", data=y_train)

        training = hf.create_group("test")
        training.create_dataset("x", data=x_test)
        training.create_dataset("y", data=y_test)

        k, v = vocab_dict_to_list(vocab_dict)
        vocab = hf.create_group("vocab_dict")
        k_utf8 = [n.encode("utf-8") for n in k]
        vocab.create_dataset("keys", data=k_utf8)
        vocab.create_dataset("values", data=v)


def process_review(text: str, regex) -> str:
    text = text.lower()
    text = text.strip()
    text = text.replace('<br />', '')
    text = regex.sub('', text)
    text = text.replace(' s ', ' is ')
    return text


def encode_dataset(dataset: List[List[str]], vocab_dict: Dict[str, int]) -> List[List[int]]:
    dataset_encoded = [[vocab_dict[s] for s in entry] for entry in dataset]
    return dataset_encoded


def parse_files(root: str, regex, size: float = 1) -> (List[str], List[int], Set[str]):
    vocab_set = set()

    neg = os.path.join(root, 'neg')
    pos = os.path.join(root, 'pos')
    list_reviews = [os.path.join(neg, filename) for filename in os.listdir(neg)]
    list_reviews.extend([os.path.join(pos, filename) for filename in os.listdir(pos)])

    n = int(len(list_reviews) * size)
    list_reviews = list_reviews[:n]

    x = []
    y = []

    for filepath in list_reviews:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            text = process_review(text, regex)
            text = text.split(' ')
            text = list(filter(lambda a: a != '', text))
            # noinspection PyTypeChecker
            vocab_set.update(text)
            # gets the number after the _ and before the .txt
            rating = int(filepath.split('_')[1][:-4])

            x.append(text)
            y.append(rating)

    return x, y, vocab_set
