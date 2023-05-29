#!/bin/sh
python train.py -task ext -mode train -bert_data_path ../data/cnndm_xlnet/ -ext_dropout 0.1 -model_path ../models/ext -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/cnndm_xlnet_ext -use_interval true -warmup_steps 10000 -max_pos 512 -temp_dir ../XLnet/base
