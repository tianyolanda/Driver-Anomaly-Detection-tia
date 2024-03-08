export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --root_path /home/ubuntu/datasets/DADdataset/DAD/DAD/ \
  --mode train \
  --view front_depth \
  --model_type mobilenetv2se\
  --pre_train_model False \
  --n_train_batch_size 5 \
  --a_train_batch_size 40 \
  --width_mult 1.0 \
  --val_batch_size 20 \
  --learning_rate 0.001 \
  --epochs 250 \
  --cal_vec_batch_size 20 \
  --tau 0.1 \
  --train_crop 'random' \
  --n_scales 5 \
  --downsample 2 \
  --n_split_ratio 1.0 \
  --a_split_ratio 1.0 \
  --n_threads 2\
  --resume_path 'mobilenetv2se_front_depth_110.pth' \
  --resume_head_path 'mobilenetv2se_front_depth_110_head.pth' \







