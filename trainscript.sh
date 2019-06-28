python train.py \
	--train_lists ./300w_train.json \
    --eval_lists ./300w_test_full.json \
                 ./300w_test_common.json \
                 ./300w_test_challenge.json \
	--num_pts 68 \
    --save_path ./snapshots/  \
    --arg_flip \
    --nesterov
