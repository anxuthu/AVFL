import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

import argparse

parser =argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-r', '--test_ratio', type=float, default=0.2)
args = parser.parse_args()

def shuffle_split(filename, test_ratio):
    assert test_ratio > 0 and test_ratio < 1

    x, y = load_svmlight_file(filename)
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)

    split = int(x.shape[0] * test_ratio)

    test_idx = idx[:split]
    train_idx = idx[split:]

    x_train, y_train = x[train_idx, :], y[train_idx]
    x_test, y_test = x[test_idx, :], y[test_idx]

    dump_svmlight_file(x_test, y_test, filename+'_test')
    dump_svmlight_file(x_train, y_train, filename+'_train')
    print(filename, x_train.shape, x_test.shape)


if __name__ == '__main__':
    shuffle_split(args.filename, args.test_ratio)
