#!/bin/bash

./scripts/build.sh

/usr/local/openmpi-4.0.1/bin/mpirun -np 8 ./bin/afd \
	--eta 1e-4 --reg 1e-4 --seed 1234 \
	--d 4272227 --n_train 16087 --n_test 3308 \
	--inner 16087 --save_interval 16087 --save_num 5 \
	--model 1 --method 1 --sync 1 --delay_fraction 0 \
	--train_data "/export/UserData/an/regression/log1p_E2006_train" \
	--test_data "/export/UserData/an/regression/log1p_E2006_test" \
	2>&1 | tee tmp-4
