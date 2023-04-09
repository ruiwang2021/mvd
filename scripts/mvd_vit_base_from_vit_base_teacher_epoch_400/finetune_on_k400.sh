#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`
OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_k400_epoch_400/finetune_on_k400'
MODEL_PATH='OUTPUT/mvd_vit_base_with_vit_base_teacher_k400_epoch_400/checkpoint-399.pth'
DATA_PATH='k400_anno'
DATA_ROOT='your_path/kinetics400'

# train on 32 V100 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
    --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size 6 --update_freq 2 --num_sample 2 \
    --save_ckpt_freq 5 --no_save_best_ckpt \
    --num_frames 16 --sampling_rate 4 \
    --lr 5e-4 --epochs 75 \
    --dist_eval --test_num_segment 5 --test_num_crop 3 \
    --enable_deepspeed
