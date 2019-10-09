import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

import argparse

parser =argparse.ArgumentParser()
parser.add_argument('--train_data', type=str)
parser.add_argument('--test_data', type=str)
args = parser.parse_args()

def standardize(train_data, test_data):
    x_train, y_train = load_svmlight_file(train_data)
    x_test, y_test = load_svmlight_file(test_data)

    #mx = np.mean(x_train, axis=0)
    #sx = np.sqrt(np.mean(np.power(x_train - mx, 2), axis=0))#TODO
    my = np.mean(y_train)
    sy = np.std(y_train)

    #rows, cols = x_train.nonzero()
    #for row, col in zip(rows, cols):
    #    x_train[row, col] = (x_train[row, col] - mx[0, col]) / sx[0, col]
    #rows, cols = x_test.nonzero()
    #for row, col in zip(rows, cols):
    #    x_test[row, col] = (x_test[row, col] - mx[0, col]) / sx[0, col]

    y_train = (y_train - my) / sy
    y_test = (y_test - my) / sy
    dump_svmlight_file(x_train, y_train, train_data+'_std')
    dump_svmlight_file(x_test, y_test, test_data+'_std')


if __name__ == '__main__':
    standardize(args.train_data, args.test_data)
