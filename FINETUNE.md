# Fine-tuning

The implementation of our MVD supports **multi-node distributed training**. We provide the **off-the-shelf** scripts in the [scripts folder](scripts).

-  For example, to fine-tune MVD ViT-Base on **Something-Something V2** with 16 GPUs (2 nodes x 8 GPUs), you can run

  ```bash
  GPUS=8
  NODE_COUNT=2
  RANK=0
  MASTER_PORT=29500
  OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_k400_epoch_400/finetune_on_ssv2'
  MODEL_PATH='OUTPUT/mvd_vit_base_with_vit_base_teacher_k400_epoch_400/checkpoint-399.pth'
  DATA_PATH='ssv2_anno'
  DATA_ROOT='your_path/ssv2/videos'
  
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
      --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
      --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
      run_class_finetuning.py \
      --model vit_base_patch16_224 \
      --data_set SSV2 --nb_classes 174 \
      --data_path ${DATA_PATH} \
      --data_root ${DATA_ROOT} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --input_size 224 --short_side_size 224 \
      --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
      --batch_size 24 --update_freq 1 --num_sample 2 \
      --save_ckpt_freq 5 --no_save_best_ckpt \
      --num_frames 16 \
      --lr 5e-4 --epochs 30 \
      --dist_eval --test_num_segment 2 --test_num_crop 3 \
      --use_checkpoint \
      --enable_deepspeed
  ```

  We set `RANK` (`--node_rank`) as `0` on the first node. On other nodes, run the same command with `RANK=1` respectively.  `--master_addr` is set as the ip of the node 0.

- For example, to fine-tune MVD ViT-Base on **Kinetics400** with 32 GPUs (4 nodes x 8 GPUs), you can run

  ```bash
  GPUS=8
  NODE_COUNT=4
  RANK=0
  MASTER_PORT=29500
  OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_k400_epoch_400/finetune_on_k400'
  MODEL_PATH='OUTPUT/mvd_vit_base_with_vit_base_teacher_k400_epoch_400/checkpoint-399.pth'
  DATA_PATH='k400_anno'
  DATA_ROOT='your_path/kinetics400'
  
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
  ```

  We set `RANK` (`--node_rank`) as `0` on the first node. On other nodes, run the same command with `RANK=1`, ..., `RANK=3` respectively.  `--master_addr` is set as the ip of the node 0.

### Note:

- We perform the **I3D dense sampling** on **Kinetics400** and **uniform sampling** on **Something-Something V2**, respectively.
- We didn't use `cls token` in our implementation, and directly average the feature of last layer for video classification.
- Here total batch size = (`batch_size` per gpu) x `update frequency` x `nodes` x (gpus per node).
- `lr` here is the base learning rate. The ` actual lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `` actual lr`` = `lr` * total batch size / 256.
- `DATA_ROOT` is set to the root directory of the dataset if relative paths are used in the annotation of the dataset.
