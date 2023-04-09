import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial, reduce
from operator import mul
from einops import rearrange
import torch.utils.checkpoint as checkpoint

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table, get_3d_sincos_pos_embed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import drop_path, to_2tuple


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_masked_video_student_small_patch16_224',
    'pretrain_masked_video_student_base_patch16_224',
    'pretrain_masked_video_student_large_patch16_224',
    'pretrain_masked_video_student_huge_patch16_224',
]


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainMaskedVideoStudent(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_depth=4,
                 feat_decoder_embed_dim=None,
                 feat_decoder_num_heads=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 tubelet_size=2,
                 num_frames=16,
                 use_cls_token=False,
                 target_feature_dim=768,
                 target_video_feature_dim=768,
                 use_checkpoint=False,
                 ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim, tubelet_size=tubelet_size, num_frames=num_frames)
        self.patch_size = self.patch_embed.patch_size
        num_patches = self.patch_embed.num_patches
        self.encoder_embed_dim = encoder_embed_dim
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.use_checkpoint = use_checkpoint

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        else:
            self.cls_token = None

        # sine-cosine positional embeddings
        self.pos_embed = get_3d_sincos_pos_embed(embed_dim=encoder_embed_dim,
                                                 grid_size=self.patch_embed.num_patches_h,
                                                 t_size=self.patch_embed.num_patches_t)
        self.pos_embed = nn.Parameter(self.pos_embed, requires_grad=False)
        self.pos_embed.requires_grad = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=encoder_embed_dim, num_heads=encoder_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)

        if feat_decoder_embed_dim is None:
            feat_decoder_embed_dim = encoder_embed_dim
        if feat_decoder_num_heads is None:
            feat_decoder_num_heads = encoder_num_heads
        self.mask_token_img = nn.Parameter(torch.zeros(1, 1, feat_decoder_embed_dim))
        self.down_img = nn.Linear(encoder_embed_dim, feat_decoder_embed_dim)
        self.decoder_img = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_classes=target_feature_dim,
            embed_dim=feat_decoder_embed_dim,
            depth=decoder_depth,
            num_heads=feat_decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_checkpoint=use_checkpoint,
        )
        self.pos_embed_img = get_3d_sincos_pos_embed(
            embed_dim=feat_decoder_embed_dim,
            grid_size=self.patch_embed.num_patches_h,
            t_size=self.patch_embed.num_patches_t
        )
        self.pos_embed_img = nn.Parameter(self.pos_embed_img, requires_grad=False)
        self.pos_embed_img.requires_grad = False
        trunc_normal_(self.mask_token_img, std=.02)

        self.mask_token_vid = nn.Parameter(torch.zeros(1, 1, feat_decoder_embed_dim))
        self.down_vid = nn.Linear(encoder_embed_dim, feat_decoder_embed_dim)
        self.decoder_vid = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_classes=target_video_feature_dim,
            embed_dim=feat_decoder_embed_dim,
            depth=decoder_depth,
            num_heads=feat_decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_checkpoint=use_checkpoint,
        )
        self.pos_embed_vid = get_3d_sincos_pos_embed(
            embed_dim=feat_decoder_embed_dim,
            grid_size=self.patch_embed.num_patches_h,
            t_size=self.patch_embed.num_patches_t
        )
        self.pos_embed_vid = nn.Parameter(self.pos_embed_vid, requires_grad=False)
        self.pos_embed_vid.requires_grad = False
        trunc_normal_(self.mask_token_vid, std=.02)

        self.apply(self._init_weights)

        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=1e-6)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward_encoder(self, x, mask):
        # embed patches
        # x: B, C, T, H, W
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed.type_as(x).detach()
        # x: B, L, C

        # masking: length -> length * mask_ratio
        B, _, C = x.shape
        x = x[~mask].reshape(B, -1, C)  # ~mask means visible

        # append cls token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)

        return x

    def forward(self, x, mask):
        x = self.forward_encoder(x, mask)
        s = 1 if self.use_cls_token else 0

        x_vis_img = self.down_img(x)
        B, N, C = x_vis_img.shape

        expand_pos_embed_img = self.pos_embed_img.type_as(x_vis_img).detach().expand(B, -1, -1)
        pos_emd_vis_img = expand_pos_embed_img[~mask].reshape(B, -1, C)
        pos_emd_mask_img = expand_pos_embed_img[mask].reshape(B, -1, C)
        x_img = torch.cat(
            [x_vis_img[:, s:, :] + pos_emd_vis_img, self.mask_token_img + pos_emd_mask_img],
            dim=1)  # [B, N, C_d]
        x_img = torch.cat([x_vis_img[:, :s, :], x_img], dim=1)

        x_img = self.decoder_img(x_img, pos_emd_mask_img.shape[1])

        x_vis_vid = self.down_vid(x)
        B, N, C = x_vis_vid.shape

        expand_pos_embed_vid = self.pos_embed_vid.type_as(x_vis_vid).detach().expand(B, -1, -1)
        pos_emd_vis_vid = expand_pos_embed_vid[~mask].reshape(B, -1, C)
        pos_emd_mask_vid = expand_pos_embed_vid[mask].reshape(B, -1, C)
        x_vid = torch.cat(
            [x_vis_vid[:, s:, :] + pos_emd_vis_vid, self.mask_token_vid + pos_emd_mask_vid],
            dim=1)  # [B, N, C_d]
        x_vid = torch.cat([x_vis_vid[:, :s, :], x_vid], dim=1)

        x_vid = self.decoder_vid(x_vid, pos_emd_mask_vid.shape[1])

        return x_img, x_vid


@register_model
def pretrain_masked_video_student_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainMaskedVideoStudent(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_masked_video_student_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainMaskedVideoStudent(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_masked_video_student_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainMaskedVideoStudent(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_masked_video_student_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainMaskedVideoStudent(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
