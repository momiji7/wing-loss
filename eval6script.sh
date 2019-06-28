python eval6.py \
	--train_lists ./300w_train.json \
	--eval_lists ./300w_test_full.json \
                 ./300w_test_common.json \
                 ./300w_test_challenge.json \
	--num_pts 68 \
	--save_path ./snapshots/  \
	--crop_width 64 --crop_height 64 \
	--LR 0.00003  \
	--epochs 320   \
	--schedule 100 200 \
	--arg_flip \
    --nesterov \