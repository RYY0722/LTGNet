save_dir=PATH_TO_SAVE ## output: PATH_TO_SAVE/save_results/comb
model_path=MODEL_PATH  ### download the pretrained model 
dataset=DATASET_PATH  ### place the original 6x6 dataset here
use_gpu=False  ### set to True if GPU available

python main.py --save_dir $save_dir \
            --reset True \
            --log_file_name test.log \
            --num_res_blocks 4+4+2+1 \
            --test True \
            --test_gt False \
            --num_workers 4 \
            --cpu $use_gpu \  
            --dataset_dir $dataset \
            --model_path $model_path