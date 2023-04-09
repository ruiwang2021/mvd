import os
from torchvision import transforms
from transforms import *
import video_transforms
from masking_generator import TubeMaskingGenerator, RandomMaskingGenerator
from kinetics import VideoClsDataset, VideoDistillation
from ssv2 import SSVideoClsDataset


class DataAugmentationForVideoDistillation(object):
    def __init__(self, args, num_frames=None):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleTwoResizedCrop(
            args.input_size, args.teacher_input_size, [1, .875, .75, .66]
        )
        self.transform = transforms.Compose([
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        window_size = args.window_size if num_frames is None else (num_frames // args.tubelet_size, args.window_size[1], args.window_size[2])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, args.mask_ratio
            )
        elif args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data_0, process_data_1, labels = self.train_augmentation(images)
        process_data_0, _ = self.transform((process_data_0, labels))
        process_data_1, _ = self.transform((process_data_1, labels))
        return process_data_0, process_data_1, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoDistillation,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_distillation_dataset(args, num_frames=None):
    if num_frames is None:
        num_frames = args.num_frames
    transform = DataAugmentationForVideoDistillation(args, num_frames=num_frames)
    dataset = VideoDistillation(
        root=args.data_root,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        num_sample=args.num_sample,
        num_segments=args.num_sample,
    )
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 400

    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_root,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
