/home/an/third_party/bin/mpirun -np 8 ./bin/afdmc \
    --reg 1e-4 --eta 0.1 \
    --d 47236 --n_train 518571 --n_test 15564 --n_class 53 \
    --inner 518571 --save_interval 20000 --save_num 40 \
    --method 0 --sync 0 --delay_fraction 0 \
    --train_data "./datasets/rcv1_test.multiclass" \
    --test_data "./datasets/rcv1_train.multiclass"
