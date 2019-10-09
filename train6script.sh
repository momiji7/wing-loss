python train6.py \
	--train_lists ./pdb/300w_train_pdb.json \
	--eval_lists ./300w_test_full.json \
                 ./300w_test_common.json \
                 ./300w_test_challenge.json \
	--num_pts 68 \
	--save_path ./snapshots/  \
	--crop_width 64 --crop_height 64 \
	--LR 0.0003  \
	--epochs 300   \
	--schedule 100 200 \
    --nesterov \
    --decay 0.0005 \
    --gaussianblur_kernel_size 3 \
    --transbbox_percent 0.05