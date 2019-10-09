#!/bin/bash

./scripts/build.sh

/usr/local/openmpi-4.0.1/bin/mpirun -np 4 ./bin/afd \
	--eta 0.01 --reg 1e-4 --seed 1234 \
	--d 92 --n_train 96257 --n_test 24012 \
	--inner 96257 --save_interval 96257 --save_num 40 \
	--model 0 --method 1 --sync 0 --delay_fraction 1.5 \
	--train_data "/export/UserData/an/GiveMeSomeCredit/GiveMeSomeCredit_train" \
	--test_data "/export/UserData/an/GiveMeSomeCredit/GiveMeSomeCredit_test" \
	2>&1 | tee givemesomecredit0
