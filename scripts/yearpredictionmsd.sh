#!/bin/bash

./scripts/build.sh

/usr/local/openmpi-4.0.1/bin/mpirun -np 4 ./bin/afd \
	--eta 5e-9 --reg 1e-4 --seed 1234 \
	--d 90 --n_train 463715 --n_test 51630 \
	--inner 463715 --save_interval 463715 --save_num 50 \
	--model 1 --method 0 --sync 1 --delay_fraction 1.5 \
	--train_data "/export/UserData/an/regression/YearPredictionMSD_train_std" \
	--test_data "/export/UserData/an/regression/YearPredictionMSD_test_std" \
	2>&1 | tee tmp3
