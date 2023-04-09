# Data Preparation

- The pre-processing of **Kinetics400** can be summarized into 3 steps:

  1. Download the dataset from [official website](https://deepmind.com/research/open-source/kinetics).

  2. Preprocess the dataset by resizing the short edge of video to **320px**. 

  3. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations) in `k400_anno`. The annotation usually includes `train.csv`, `val.csv` and `test.csv` ( here `test.csv` is the same as `val.csv`). The format of `*.csv` file is like:

     ```
     dataset_root/video_1.mp4  label_1
     dataset_root/video_2.mp4  label_2
     dataset_root/video_3.mp4  label_3
     ...
     dataset_root/video_N.mp4  label_N
     ```


- The pre-processing of **Something-Something-V2** can be summarized into 3 steps:

  1. Download the dataset from [official website](https://developer.qualcomm.com/software/ai-datasets/something-something).

  2. (optional) Preprocess the dataset by changing the video extension from `webm` to `.mp4` with the **original** height of **240px**.

  3. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations) in `ssv2_anno`. The annotation usually includes `train.csv`, `val.csv` and `test.csv` ( here `test.csv` is the same as `val.csv`). The format of `*.csv` file is like:

   ```
   dataset_root/video_1.webm  label_1
   dataset_root/video_2.webm  label_2
   dataset_root/video_3.webm  label_3
   ...
   dataset_root/video_N.webm  label_N
   ```
