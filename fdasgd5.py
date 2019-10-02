"""numpy is row-major in default"""
import sys
import os
import copy

import argparse
import time

import numpy as np
from sklearn.datasets import load_svmlight_file
from logistic import *

import zmq
from mpi4py import MPI

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Array, Queue


parser = argparse.ArgumentParser()
parser.add_argument('-t', default=100, type=int, help='#iterations')
parser.add_argument('--reg', default=1e-4, type=float, help='regularization')
parser.add_argument('--eta', default=100, type=float, help='step size')

parser.add_argument('--train_data', type=str,
        default='/export/UserData/an/real-sim')
parser.add_argument('--test_data', type=str,
        default='/export/UserData/an/real-sim.t')
parser.add_argument('-d', type=int, default=20958, help='#features')
parser.add_argument('--n_train', type=int, default=57848)
parser.add_argument('--n_test', type=int, default=14461)

parser.add_argument('-si', '--save_interval', type=float, default=0.01,
        help='save every seconds')
parser.add_argument('-sn', '--save_num', type=int, default=500,
        help='total save number for evaluation')

parser.add_argument('-bp', '--base_port', type=int, default=49152)

args = parser.parse_args()
print(args, flush=True)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
np.random.seed(123)#TODO

""" allocate features [start_feature, end_feature) """
feature_sizes = [args.d // size for i in range(size)]
for i in range(args.d % size):
    feature_sizes[i] += 1
start_feature = np.sum(feature_sizes[:rank], dtype=int)
end_feature = start_feature + feature_sizes[rank]

""" shared memory """
#(xl, yl) = load_svmlight_file(args.train_data, n_features=args.d, dtype=float)
#assert (args.n_train, args.d) == xl.shape
#xl, yl = (xl.toarray()[:, start_feature:end_feature], yl[:])

wlshm = Array('d', feature_sizes[rank], lock=True)
main_q = Queue()
sender_q = Queue()


""" Training Process """
def main():
    """ initialize variables """
    wl = np.frombuffer(wlshm.get_obj())

    """ training """
    for t in range(2):
        print(rank, t)
        idx = 233

        # total prod
        sender_q.put(idx)
        total_prod = main_q.get()

    print(rank)
    while True:
        pass


def get_ip(rank, shift=0, base_port=args.base_port):
    port = rank + shift + base_port
    return "tcp://150.212.182.43:" + str(port)


# message: [rank, idx]
def sender():
    context = zmq.Context()
    s_socket = context.socket(zmq.PUSH)

    r_socket = context.socket(zmq.PULL)
    r_socket.bind(get_ip(rank=rank, shift=0))

    while True:
        idx = sender_q.get()
        for r in range(size):
            if r != rank:
                s_socket.connect(get_ip(rank=r, shift=size))
                s_socket.send_pyobj([rank, idx])

        total_prod = 0
        for r in range(size - 1):
            total_prod += r_socket.recv_pyobj()
        main_q.put(total_prod)


# message: [rank, idx]
def receiver():
    context = zmq.Context()
    s_socket = context.socket(zmq.PUSH)

    r_socket = context.socket(zmq.PULL)
    r_socket.bind(get_ip(rank=rank, shift=size))

    while True:
        r, idx = r_socket.recv_pyobj()
        #wl = np.frombuffer(wlshm.get_obj())
        #with wlshm.get_lock():
        #    prod = xl[idx].dot(wl)
        prod = 0
        s_socket.connect(get_ip(rank=r, shift=0))
        s_socket.send_pyobj(prod)


if __name__ == '__main__':
    pe = ProcessPoolExecutor(max_workers=5)
    pe.submit(main)
    pe.submit(sender)
    pe.submit(receiver)
    pe.shutdown()
