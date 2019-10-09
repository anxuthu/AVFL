#!/bin/bash

./scripts/build.sh

/usr/local/openmpi-4.0.1/bin/mpirun -np 4 ./bin/afd \
	--eta 0.01 --reg 1e-4 --seed 1234 \
	--d 90 --n_train 24000 --n_test 6000 \
	--inner 24000 --save_interval 24000 --save_num 100 \
	--model 0 --method 1 --sync 0 --delay_fraction 1 \
	--train_data "/export/UserData/an/UCI_Credit_Card/UCI_Credit_Card_train" \
	--test_data "/export/UserData/an/UCI_Credit_Card/UCI_Credit_Card_test" \
	2>&1 | tee uci0
