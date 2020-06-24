DATA_DIR=~/data
SAVE_CSV="pred.csv"
LOAD_CKPT="./ckpt/best.pth"

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --test --data_dir $DATA_DIR --test_out_csv_path $SAVE_CSV \
                         --load_ckpt $LOAD_CKPT
