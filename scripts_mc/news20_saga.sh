/home/an/third_party/bin/mpirun -np 8 ./bin/afdmc \
    --reg 1e-4 --eta 0.3 \
    --d  62061 --n_train 15935 --n_test 3993 --n_class 20 \
    --inner 15935 --save_interval 15935 --save_num 50 \
    --method 2 --sync 1 --delay_fraction 0 \
    --train_data "./datasets/news20.scale" \
    --test_data "./datasets/news20.t.scale"
