export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --root_path /home/ubuntu/datasets/DADdataset/DAD/DAD/ \
  --mode train \
  --view front_depth \
  --model_type mobilenetv2\
  --pre_train_model False \
  --n_train_batch_size 3 \
  --a_train_batch_size 25 \
  --val_batch_size 20 \
  --learning_rate 0.01 \
  --epochs 250 \
  --cal_vec_batch_size 100 \
  --tau 0.1 \
  --train_crop 'random' \
  --n_scales 5 \
  --downsample 2 \
  --n_split_ratio 1.0 \
  --a_split_ratio 1.0 \
  --n_threads 0\
  --resume_path 'mobilenetv2_front_depth_180.pth' \
  --resume_head_path 'mobilenetv2_front_depth_180_head.pth' \






