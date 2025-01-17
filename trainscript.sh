CUDA_VISIBLE_DEVICES=6,7 python train.py \
	--train_lists ./pdb/300w_train_pdb.json \
	--eval_lists ./300w_test_full.json \
                 ./300w_test_common.json \
                 ./300w_test_challenge.json \
	--num_pts 68 \
	--save_path ./snapshots/  \
	--crop_width 224 --crop_height 224 \
	--LR 0.00003  \
	--epochs 320   \
	--schedule 200 300 \
    --decay 0.0005 \
    --nesterov 

