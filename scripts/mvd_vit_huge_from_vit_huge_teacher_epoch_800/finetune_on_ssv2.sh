#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`
OUTPUT_DIR='OUTPUT/mvd_vit_huge_with_vit_huge_teacher_k400_epoch_800/finetune_on_ssv2'
MODEL_PATH='OUTPUT/mvd_vit_huge_with_vit_huge_teacher_k400_epoch_800/checkpoint-799.pth'
DATA_PATH='ssv2_anno'
DATA_ROOT='your_path/ssv2/videos'

# train on 16 V100 GPUs (2 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
    --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model vit_huge_patch16_224 \
    --data_set SSV2 --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size 8 --update_freq 1 --num_sample 2 \
    --save_ckpt_freq 5 --no_save_best_ckpt \
    --num_frames 16 \
    --lr 5e-4 --min_lr 1e-5 --epochs 30 \
    --fc_drop_rate 0.5 --drop_path 0.3 \
    --dist_eval --test_num_segment 2 --test_num_crop 3 \
    --use_checkpoint \
    --enable_deepspeed
