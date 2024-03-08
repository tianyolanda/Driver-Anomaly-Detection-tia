export PYTHONPATH=$PWD
python main-onemode-readscore.py \
  --root_path /home/ubuntu/datasets/DADdataset/DAD/DAD/ \
  --mode test \
  --view front_depth \
  --model_type resnet \
  --model_depth 18 \
  --shortcut_type A \
  --pre_train_model False \
  --n_train_batch_size 10 \
  --a_train_batch_size 150 \
  --val_batch_size 70\
  --learning_rate 0.01 \
  --epochs 250 \
  --cal_vec_batch_size 100 \
  --tau 0.1 \
  --resume_path 'resnet18_frontdepth/best_model_resnet_front_depth.pth' \
  --resume_head_path 'resnet18_frontdepth/best_model_resnet_front_depth_head.pth' \
  --train_crop 'random' \
  --n_scales 5 \
  --downsample 2 \
  --n_split_ratio 1.0 \
  --a_split_ratio 1.0 \






