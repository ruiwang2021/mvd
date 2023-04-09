#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`
OUTPUT_DIR='OUTPUT/mvd_vit_large_with_vit_large_teacher_k400_epoch_800/finetune_on_k400'
MODEL_PATH='OUTPUT/mvd_vit_large_with_vit_large_teacher_k400_epoch_800/checkpoint-799.pth'
DATA_PATH='k400_anno'
DATA_ROOT='your_path/kinetics400'

# train on 32 V100 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
    --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size 16 --update_freq 1 --num_sample 2 \
    --save_ckpt_freq 5 --no_save_best_ckpt \
    --num_frames 16 --sampling_rate 4 \
    --lr 1e-3 --epochs 50 \
    --fc_drop_rate 0.5 --drop_path 0.3 \
    --dist_eval --test_num_segment 5 --test_num_crop 3 \
    --use_checkpoint \
    --enable_deepspeed
