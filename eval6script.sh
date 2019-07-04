python eval6.py \
	--train_lists ./qumeng.json \
	--eval_lists ./qumeng.json \
	--num_pts 68 \
	--save_path ./snapshots/  \
	--crop_width 64 --crop_height 64 \
	--LR 0.00003  \
	--epochs 320   \
	--schedule 100 200 \
    --nesterov \