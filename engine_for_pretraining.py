import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import torchvision
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

Loss_func_choice = {'L1': torch.nn.L1Loss, 'L2': torch.nn.MSELoss, 'SmoothL1': torch.nn.SmoothL1Loss}


def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, update_freq=None, time_stride_loss=True, lr_scale=1.0,
                    image_teacher_model=None, video_teacher_model=None, norm_feature=False):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    LN_img = nn.LayerNorm(args.distillation_target_dim, eps=1e-6, elementwise_affine=False).cuda()
    LN_vid = nn.LayerNorm(args.video_distillation_target_dim, eps=1e-6, elementwise_affine=False).cuda()

    loss_func_img_feat = Loss_func_choice[args.distill_loss_func]()
    loss_func_vid_feat = Loss_func_choice[args.video_distill_loss_func]()
    image_loss_weight = args.image_teacher_loss_weight
    video_loss_weight = args.video_teacher_loss_weight

    tubelet_size = args.tubelet_size

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        update_step = step // update_freq
        it = start_steps + update_step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None and step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"] * lr_scale
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, videos_for_teacher, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        videos_for_teacher = videos_for_teacher.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        _, _, T, _, _ = videos.shape

        with torch.cuda.amp.autocast():
            output_features, output_video_features = model(videos, bool_masked_pos)
            with torch.no_grad():
                image_teacher_model.eval()
                if time_stride_loss:
                    teacher_features = image_teacher_model(
                        rearrange(videos_for_teacher[:, :, ::tubelet_size, :, :], 'b c t h w -> (b t) c h w'),
                    )
                    teacher_features = rearrange(teacher_features, '(b t) l c -> b (t l) c', t=T//tubelet_size)
                else:
                    teacher_features = image_teacher_model(
                        rearrange(videos_for_teacher, 'b c t h w -> (b t) c h w'),
                    )
                    teacher_features = rearrange(teacher_features, '(b t d) l c -> b (t l) (d c)', t=T//tubelet_size, d=tubelet_size)
                if norm_feature:
                    teacher_features = LN_img(teacher_features)

                video_teacher_model.eval()
                videos_for_video_teacher = videos if args.video_teacher_input_size == args.input_size \
                    else videos_for_teacher

                video_teacher_features = video_teacher_model(videos_for_video_teacher)
                if norm_feature:
                    video_teacher_features = LN_vid(video_teacher_features)

            B, _, D = output_features.shape
            loss_img_feat = loss_func_img_feat(
                input=output_features,
                target=teacher_features[bool_masked_pos].reshape(B, -1, D)
            )
            loss_value_img_feat = loss_img_feat.item()

            B, _, D = output_video_features.shape
            loss_vid_feat = loss_func_vid_feat(
                input=output_video_features,
                target=video_teacher_features[bool_masked_pos].reshape(B, -1, D)
            )
            loss_value_vid_feat = loss_vid_feat.item()

            loss = image_loss_weight * loss_img_feat + video_loss_weight * loss_vid_feat

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(step + 1) % update_freq == 0)
        if (step + 1) % update_freq == 0:
            optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_img_feat=loss_value_img_feat)
        metric_logger.update(loss_vid_feat=loss_value_vid_feat)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_img_feat=loss_value_img_feat, head="loss_img_feat")
            log_writer.update(loss_vid_feat=loss_value_vid_feat, head="loss_vid_feat")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
