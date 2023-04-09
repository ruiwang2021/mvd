# Pre-training

The implementation of our MVD supports **multi-node distributed training**. We provide the **off-the-shelf** scripts in the [scripts folder](scripts).

For convenience, we use publicly available pretrained models from [MAE](https://github.com/facebookresearch/mae) and [VideoMAE](https://github.com/MCG-NJU/VideoMAE) as the teacher models.
- Before pretraining, you can download pretrained checkpoints from the repo of [MAE](https://github.com/facebookresearch/mae) and [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md).
- For example, you can download the [ViT-Base image teacher](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and the [ViT-Base video teacher](https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1/view?usp=sharing). Then rename them for convenience.

- For example, to pre-train MVD ViT-Base with ViT-Base teachers on **Kinetics400** with 32 GPUs (4 nodes x 8 GPUs), you can run

  ```bash
  GPUS=8
  NODE_COUNT=4
  RANK=0
  MASTER_PORT=29500
  OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_k400_epoch_400'
  DATA_PATH='k400_anno/train.csv'
  DATA_ROOT='your_path/kinetics400'

  OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS} \
          --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
          --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
          run_mvd_pretraining.py \
          --data_path ${DATA_PATH} \
          --data_root ${DATA_ROOT} \
          --model pretrain_masked_video_student_base_patch16_224 \
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
  ```

  We set `RANK` (`--node_rank`) as `0` on the first node. On other nodes, run the same command with `RANK=1`, ..., `RANK=3` respectively.  `--master_addr` is set as the ip of the node 0.

### Note:

- Here the batch size is 16 (`batch_size` per gpu) * 2 (`update frequency`) * 4 (`nodes`) * 8 (gpus per node) = 1024.
- `lr` here is the base learning rate and is set to `1.5e-4` as default. The ` actual lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `` actual lr`` = `lr` * total batch size / 256.
- `DATA_ROOT` is set to the root directory of the dataset if relative paths are used in the annotation of the dataset.
