CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python cxr_main.py \
  --dim 256 --dropout 0.3 --layers 2 \
  --mode train \
  --epochs 50 \
  --batch_size 16 \
  --data_pairs readmission \
  --pretrained \
  --vision-backbone resnet34 \
  --vision_num_classes 1 \
  --num_classes 1 \
  --labels_set readmission \
  --load_state './official_pretrained_weight/best_checkpoint.pth.tar'\
  --save_dir checkpoints/cxr/readmission/superod_01
