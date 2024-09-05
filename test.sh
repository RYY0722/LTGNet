save_dir="testing"
model_path="logs-0.3/model/model_best_psnr.pth"
dataset="data\test_w_gt"
# dataset="test_wo_gt"

python main.py --save_dir $save_dir \
            --reset True \
            --log_file_name test.log \
            --num_res_blocks 4+4+2+1 \
            --test True \
            --test_gt True \
            --num_workers 4 \
            --dataset_dir $dataset \
            --model_path $model_path