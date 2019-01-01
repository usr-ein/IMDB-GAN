import json
import os
from sys import argv

from matplotlib import pyplot as plt


def main(filename=None):
    while True:
        if filename is None:
            filename = ask_filename()

        metrics = load_metrics(filename)
        export(metrics, filename)

        if len(filename) == len([f for f in os.listdir('data/metrics') if '.json' in f]):
            break
        resp = input('Another graph ? (y/n) ')
        if 'y' not in resp:
            break
        filename = None


def load_metrics(filename):
    list_metrics = []
    for fn in filename:
        with open(fn) as json_data:
            list_metrics.append(json.load(json_data))
            json_data.close()

    return list_metrics


def export(metrics, filename):
    for metric, fn in zip(metrics, filename):
        if 'd' in metric and 'g' in metric:
            # save_filename = fn[:-5] + '.png' # Saves in data/
            save_filename = 'graphs' + fn[fn.rfind('/'):-5] + '.png'  # Saves in graphs/
            graph(metric['d'], metric['g'], save_filename=save_filename)


def ask_filename():
    list_files = [fn for fn in os.listdir('data/metrics') if '.json' in fn]
    list_files.append('all of them')
    print("Please select metrics file amongst:")
    for fn, i in zip(list_files, range(len(list_files))):
        print('({}) {}'.format(i, fn))
    index = input('Choice (type digit): ')
    while not index.isdigit() or int(index) < 0 or int(index) >= len(list_files):
        print('Please enter a valid value')
        index = input('Choice (type digit): ')
    index = int(index)

    if list_files[index] == list_files[-1]:
        return ['data/metrics/' + fn for fn in list_files[:-1]]

    return ['data/metrics/' + list_files[index]]


def graph(discriminative_model_loss, gan_model_loss, save_filename):
    max_epoch = len(discriminative_model_loss)
    plt.plot(range(max_epoch), discriminative_model_loss)
    plt.plot(range(max_epoch), gan_model_loss)
    plt.title('GAN and discriminative model loss over epochs (max:{})'.format(max_epoch))
    plt.savefig(save_filename)
    plt.show()


if __name__ == '__main__':
    if len(argv) > 1:
        main(argv[1])
    else:
        main()
