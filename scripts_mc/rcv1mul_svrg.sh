/home/an/third_party/bin/mpirun -np 8 ./bin/afdmc \
    --reg 1e-4 --eta 0.01 \
    --d 47236 --n_train 518571 --n_test 15564 --n_class 53 \
    --inner 518571 --save_interval 518571 --save_num 20 \
    --method 1 --sync 1 --delay_fraction 2 \
    --train_data "./datasets/rcv1_test.multiclass" \
    --test_data "./datasets/rcv1_train.multiclass"
