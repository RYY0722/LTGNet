#!/bin/sh
source /home/ruanyuyan/.bashrc
SCRIPT_DIR=$(cd "$(dirname "$0")";pwd)
cd $SCRIPT_DIR
pwd
dataset=/home/ruanyuyan/octa/data/160s65-wref
cuda_id=5
w=0.01
name=$w
save_dir="logs/$w-2"
CUDA_VISIBLE_DEVICES=$cuda_id python main.py --save_dir $save_dir \
        --ref_w $w \
        --reset True  \
        --log_file_name train.log\
        --num_gpu 1  \
        --num_workers 3\
        --batch_size 3 \
        --train_crop_size 160\
        --num_res_blocks  4+4+2+1  \
        --dataset_dir $dataset \
        --n_feats 64\
        --lr_rate 5e-5  \
        --lr_rate_dis 5e-5 \
        --lr_rate_lte 1e-6    \
        --rec_w 1  \
        --per_w 0.01  \
        --adv_w 0.001 \
        --tpl_w 0.01 \
        --num_init_epochs 0 \
        --num_epochs 200  \
        --print_every 400  \
        --save_every 9999 \
        --val_every 1  \
        --decay 100 \
        --gamma 0.7  \
        --tpl_use_S True 
