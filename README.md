# mlopt

Sample Usage
python main.py --dataset cifar --model weight_norm --loss MSELoss --opt sgd --lr 0.0001 --epoch 4000 --analysis loss weight_norm --batch_size 512 --width 2048 --init_mode "O(1/sqrt{m})" --basis_var 0.02 --wn_scale 1 --multiple_run 1

python main.py --dataset spurious --model 2-mlp-sim-ln --loss MSELoss --opt sam --lr 0.01 --epoch 4000 --analysis loss eigs adv_eigs --batch_size 64 --sp_train_size 512 --sp_feat_dim 30 --sam_rho 0.05 --adv_eta 0.01

python main.py --dataset cifar --model WideResNet --loss CrossEntropyLoss --opt sam --lr 0.01 --epoch 200 --analysis loss eigs adv_eigs --log_interval 10 --batch_size 512 --sam_rho 0.05 --momentum 0.9 --adv_eta 0.01

python hugging_face_main.py --model_name_or_path google-bert/bert-base-cased --task_name mrpc --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 128 --learning_rate 2e-5 --optim sgd --num_train_epochs 200 --analysis_interval 10 --save_strategy no --overwrite_output_dir

General:
1. implement gd
2. when --debug, do not save directory
3. contraction of arguments to resolve dependecy problems
4. integrate language tasks

EoS Thread: 
1. average of weights from different runs
2. multiple runs for each lr, batch_size