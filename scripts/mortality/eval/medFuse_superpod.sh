# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
# --dim 256 --dropout 0.3 --layers 2 \
# --vision-backbone resnet34 \
# --mode eval \
# --epochs 50 --batch_size 16 \
# --vision_num_classes 14 --num_classes 1 \
# --data_pairs partial_ehr_cxr \
# --data_ratio 1.0 \
# --task in-hospital-mortality \
# --labels_set mortality \
# --fusion_type lstm \
# --save_dir checkpoints/mortality/medFuse

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python cxr_main.py \
  --dim 256 --dropout 0.3 --layers 2 \
  --mode eval \
  --epochs 50 \
  --batch_size 16 \
  --data_pairs mortality \
  --pretrained \
  --vision-backbone resnet34 \
  --vision_num_classes 1 \
  --num_classes 1 \
  --labels_set mortality \
  --load_state './official_pretrained_weight/best_checkpoint.pth.tar'\
  --save_dir checkpoints/cxr/mortality/superod_01
