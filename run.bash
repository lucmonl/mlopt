#!/bin/bash

for i in 512 1024 2048
do
    python main.py --dataset mnist --model weight_norm_v2 --loss MSELoss --opt gd --lr 0.0005 --epoch 500 --analysis loss weight_norm --batch_size 512 --width $i --init_mode "O(1)" --basis_var 5.0 --wn_scale 2.0 --log_interval 10
done