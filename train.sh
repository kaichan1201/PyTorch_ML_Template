DATA_DIR=~/data
SAVE_DIR=./ckpt
# LOAD_CKPT="./ckpt/best.pth"

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --train --data_dir $DATA_DIR \
                         -e 100 --val_epoch 1 -b 128 --lr 2e-4 --num_workers 4 \
		         --optim adam --scheduler poly --warm_up_epoch 5 \
                         --save_dir $SAVE_DIR
