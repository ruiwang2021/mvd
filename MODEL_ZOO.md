# MVD Model Zoo

### Kinetics-400

| Method | Backbone | Teacher | Epoch | \#Frame |                                                  Pre-train                                                   |                             Fine-tune                             | Top-1 | Top-5 |
|:------:|:--------:|:-------:|:-----:| :-----: |:------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------:|:-----:|:-----:|
|  MVD   |  ViT-S   |  ViT-B  |  400  | 16x5x3  | [script](scripts/mvd_vit_small_from_vit_base_teacher_epoch_400/pretrain_mvd_small_on_k400.sh)/[checkpoint](https://drive.google.com/file/d/1ip91KtdrvELVJseyDKFT0ERgNP1Ykusf/view?usp=sharing) | [script](scripts/mvd_vit_small_from_vit_base_teacher_epoch_400/finetune_on_k400.sh)  | 80.6  | 94.7  |
|  MVD   |  ViT-S   |  ViT-L  |  400  | 16x5x3  |        [script](scripts/mvd_vit_small_from_vit_large_teacher_epoch_400/pretrain_mvd_small_on_k400.sh)/[checkpoint](https://drive.google.com/file/d/1HqvGxx7_JYO5JKvRT0giesl-p-Iaaesa/view?usp=sharing)        | [script](scripts/mvd_vit_small_from_vit_large_teacher_epoch_400/finetune_on_k400.sh) | 81.0  | 94.8  |
|  MVD   |  ViT-B   |  ViT-B  |  400  | 16x5x3  |         [script](scripts/mvd_vit_base_from_vit_base_teacher_epoch_400/pretrain_mvd_base_on_k400.sh)/[checkpoint](https://drive.google.com/file/d/1UGVWWYnbot9RSYMnNz8obnQ4T6PKWu5-/view?usp=sharing)          |  [script](scripts/mvd_vit_base_from_vit_base_teacher_epoch_400/finetune_on_k400.sh)  | 82.7  | 95.4  |
|  MVD   |  ViT-B   |  ViT-L  |  400  | 16x5x3  |         [script](scripts/mvd_vit_base_from_vit_large_teacher_epoch_400/pretrain_mvd_base_on_k400.sh)/[checkpoint](https://drive.google.com/file/d/1MFgpUuHTtFMiormgenmxypU9gqRDbbwb/view?usp=sharing)         | [script](scripts/mvd_vit_base_from_vit_large_teacher_epoch_400/finetune_on_k400.sh)  | 83.4  | 95.8  |
|  MVD   |  ViT-L   |  ViT-L  |  400  | 16x5x3  |        [script](scripts/mvd_vit_large_from_vit_large_teacher_epoch_400/pretrain_mvd_large_on_k400.sh)/[checkpoint](https://drive.google.com/file/d/1Zr3ayWxlauFm1bVfMzxaQIzfYvDXhMW-/view?usp=sharing)        | [script](scripts/mvd_vit_large_from_vit_large_teacher_epoch_400/finetune_on_k400.sh) | 86.0  | 96.9  |
|  MVD   |  ViT-L   |  ViT-L  |  800  | 16x5x3  |        [script](scripts/mvd_vit_large_from_vit_large_teacher_epoch_800/pretrain_mvd_large_on_k400.sh)        | [script](scripts/mvd_vit_large_from_vit_large_teacher_epoch_800/finetune_on_k400.sh) | 86.4  | 97.0  |
|  MVD   |  ViT-H   |  ViT-H  |  800  | 16x5x3  |         [script](scripts/mvd_vit_huge_from_vit_huge_teacher_epoch_800/pretrain_mvd_huge_on_k400.sh)          |  [script](scripts/mvd_vit_huge_from_vit_huge_teacher_epoch_800/finetune_on_k400.sh)  | 87.3  | 97.4  |

### Something-Something V2

| Method | Backbone | Teacher | Epoch | \#Frame |                                      Fine-tune                                       | Top-1 | Top-5 |
|:------:|:--------:|:-------:|:-----:|:-------:|:------------------------------------------------------------------------------------:|:-----:|:-----:|
|  MVD   |  ViT-S   |  ViT-B  |  400  | 16x2x3  | [script](scripts/mvd_vit_small_from_vit_base_teacher_epoch_400/finetune_on_ssv2.sh)  | 70.7  | 92.6  |
|  MVD   |  ViT-S   |  ViT-L  |  400  | 16x2x3  | [script](scripts/mvd_vit_small_from_vit_large_teacher_epoch_400/finetune_on_ssv2.sh) | 70.9  | 92.8  |
|  MVD   |  ViT-B   |  ViT-B  |  400  | 16x2x3  |  [script](scripts/mvd_vit_base_from_vit_base_teacher_epoch_400/finetune_on_ssv2.sh)  | 72.5  | 93.6  |
|  MVD   |  ViT-B   |  ViT-L  |  400  | 16x2x3  | [script](scripts/mvd_vit_base_from_vit_large_teacher_epoch_400/finetune_on_ssv2.sh)  | 73.7  | 94.0  |
|  MVD   |  ViT-L   |  ViT-L  |  400  | 16x2x3  | [script](scripts/mvd_vit_large_from_vit_large_teacher_epoch_400/finetune_on_ssv2.sh) | 76.1  | 95.4  |
|  MVD   |  ViT-L   |  ViT-L  |  800  | 16x2x3  | [script](scripts/mvd_vit_large_from_vit_large_teacher_epoch_800/finetune_on_ssv2.sh) | 76.7  | 95.5  |
|  MVD   |  ViT-H   |  ViT-H  |  800  | 16x2x3  |  [script](scripts/mvd_vit_huge_from_vit_huge_teacher_epoch_800/finetune_on_ssv2.sh)  | 77.3  | 95.7  |


### Note:

- We report the results of MVD finetuned with `I3D dense sampling` on **Kinetics400** and `TSN uniform sampling` on **Something-Something V2**, respectively.
- \#Frame = #input_frame x #clip x #crop.
- \#input_frame means how many frames are input for model during the test phase.
- \#crop means spatial crops (e.g., 3 for left/right/center crop).
- \#clip means temporal clips (e.g., 5 means repeted temporal sampling five clips with different start indices).