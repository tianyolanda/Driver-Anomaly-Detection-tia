export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0,1 python main-onemode-fir.py \
  --root_path /home/ubuntu/datasets/DADdataset/DAD/DAD/ \
  --mode test \
  --view front_IR \
  --model_type resnetse\
  --model_depth 18 \
  --shortcut_type A \
  --pre_train_model False \
  --n_train_batch_size 8 \
  --a_train_batch_size 120 \
  --val_batch_size 50\
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




