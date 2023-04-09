#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`
OUTPUT_DIR='OUTPUT/mvd_vit_small_with_vit_base_teacher_k400_epoch_400'
DATA_PATH='k400_anno/train.csv'
DATA_ROOT='your_path/kinetics400'

# train on 32 V100 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS} \
        --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
        --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
        run_mvd_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_root ${DATA_ROOT} \
        --model pretrain_masked_video_student_small_patch16_224 \
        --opt adamw --opt_betas 0.9 0.95 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --image_teacher_model mae_teacher_vit_base_patch16 \
        --distillation_target_dim 768 \
        --distill_loss_func SmoothL1 \
        --image_teacher_model_ckpt_path 'your_path/mae_pretrain_vit_base.pth' \
        --video_teacher_model pretrain_videomae_teacher_base_patch16_224 \
        --video_distillation_target_dim 768 \
        --video_distill_loss_func SmoothL1 \
        --video_teacher_model_ckpt_path 'your_path/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600.pth' \
        --mask_type tube --mask_ratio 0.9 --decoder_depth 2 \
        --batch_size 16 --update_freq 2 --save_ckpt_freq 10 \
        --num_frames 16 --sampling_rate 4 \
        --lr 1.5e-4 --min_lr 1e-4 --drop_path 0.1 --warmup_epochs 40 --epochs 401 \
        --auto_resume
