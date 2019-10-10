#!/bin/bash

./scripts/build.sh

/usr/local/openmpi-4.0.1/bin/mpirun -np 4 ./bin/afd \
	--eta 0.005 --reg 1e-4 --seed 1234 \
	--d 150360 --n_train 16087 --n_test 3308 \
	--inner 16087 --save_interval 16087 --save_num 30 \
	--model 1 --method 0 --sync 1 --delay_fraction 1.5 \
	--train_data "/export/UserData/an/regression/E2006_train" \
	--test_data "/export/UserData/an/regression/E2006_test" \
	2>&1 | tee sgd
