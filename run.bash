#!/bin/bash

python main.py --dataset cifar --model weight_norm --loss MSELoss --opt sgd --lr 0.0001 --epoch 4000 --analysis loss weight_norm --batch_size 512 --width 4096 --init_mode "O(1/sqrt{m})" --basis_var 0.02 --wn_scale 1 --momentum 0.9 --multiple_run 1
python main.py --dataset cifar --model weight_norm --loss MSELoss --opt sgd --lr 0.0001 --epoch 4000 --analysis loss weight_norm --batch_size 512 --width 4096 --init_mode "O(1/sqrt{m})" --basis_var 0.02 --wn_scale 1 --momentum 0.9 --multiple_run 1
python main.py --dataset cifar --model weight_norm --loss MSELoss --opt sgd --lr 0.0001 --epoch 4000 --analysis loss weight_norm --batch_size 512 --width 1024 --init_mode "O(1/sqrt{m})" --basis_var 0.02 --wn_scale 1 --momentum 0.9 --multiple_run 1
python main.py --dataset cifar --model weight_norm --loss MSELoss --opt sgd --lr 0.0001 --epoch 4000 --analysis loss weight_norm --batch_size 512 --width 2048 --init_mode "O(1/sqrt{m})" --basis_var 0.02 --wn_scale 1 --momentum 0.9 --multiple_run 1
python main.py --dataset cifar --model weight_norm --loss MSELoss --opt sgd --lr 0.0001 --epoch 4000 --analysis loss weight_norm --batch_size 512 --width 512 --init_mode "O(1/sqrt{m})" --basis_var 0.02 --wn_scale 1 --momentum 0.9 --multiple_run 1
