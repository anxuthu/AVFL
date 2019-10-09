#!/bin/bash

./scripts/build.sh

/usr/local/openmpi-4.0.1/bin/mpirun -np 8 ./bin/afd \
	--eta 0.005 --reg 1e-4 --seed 1234 \
	--d 3231961 --n_train 1916904 --n_test 479226 \
	--inner 1916904 --save_interval 1916904 --save_num 1 \
	--model 0 --method 0 --sync 0 --delay_fraction 2 \
	--train_data "/export/UserData/an/url_combined_train" \
	--test_data "/export/UserData/an/url_combined_test"
