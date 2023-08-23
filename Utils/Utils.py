import numpy as np
import torch
from PIL.Image import Resampling
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
from torchvision.transforms import transforms
import torch.nn.functional as F
import math


def build_transform(is_train, image_size, color_jitter=None, aa='rand-m9-mstd0.5-inc1', reprob=0., remode='const', recount=1):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation='bicubic',
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if image_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(image_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=Resampling.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def cosine_scheduler(base_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, final_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def get_params_groups_layerWiseLrDecay(model, layerWiseLrDecay, weight_decay):
    grouped_parameters = []
    lr_scale = 1
    layer_names = []
    no_WD = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        if name.endswith(".bias") or len(param.shape) == 1:
            no_WD.append(name)
        layer_names.append(name)
    layer_names.reverse()

    prev_group_name = layer_names[0].split('.')[0]
    for idx, name in enumerate(layer_names):
        if name.startswith("model.blocks"):
            cur_group_name = name.split('.')[2]
            if cur_group_name != prev_group_name:
                lr_scale = lr_scale * layerWiseLrDecay
            prev_group_name = cur_group_name
            if name in no_WD:
                print('#'*30+'activating layer wise lr decay')
                grouped_parameters += [{"params": [p for n, p in model.named_parameters() if n == name and p.requires_grad], "lr_scale": lr_scale, 'weight_decay': 0.}]
            else:
                grouped_parameters += [{"params": [p for n, p in model.named_parameters() if n == name and p.requires_grad], "lr_scale": lr_scale, 'weight_decay': weight_decay}]
        else:
            if name in no_WD:
                grouped_parameters += [{"params": [p for n, p in model.named_parameters() if n == name and p.requires_grad], "lr_scale": 1, 'weight_decay': 0.}]
            else:
                grouped_parameters += [{"params": [p for n, p in model.named_parameters() if n == name and p.requires_grad], "lr_scale": 1, 'weight_decay': weight_decay}]
    return grouped_parameters


def adjust_learning_rate(optimizer, epoch, lr, warmup_epochs, epochs, min_lr=0):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets: torch.Tensor, n_classes: int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device).fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


def Shuffle(mask_ratio, n_patches):
    n_remaining_patches = int(n_patches * (1 - mask_ratio))
    idx = torch.randperm(n_patches)
    remaining_index = idx[:n_remaining_patches]
    mask_index = idx[n_remaining_patches:]
    return remaining_index, mask_index


def interpolate_positional_encoding(positional_encoding: torch.Tensor, new_num_patches: int):
    # 获取原始位置编码的形状
    cls_encoding = positional_encoding[:, 0, :].unsqueeze(0)
    positional_encoding = positional_encoding[:, 1:, :]
    b, n, d = positional_encoding.shape

    # 计算新的图像大小
    new_image_size = int((new_num_patches) ** 0.5)

    # 将位置编码转换为图像形式
    positional_encoding = positional_encoding.reshape(b, d, int(n ** 0.5), int(n ** 0.5))

    # 对位置编码进行插值
    positional_encoding = F.interpolate(positional_encoding, size=new_image_size, mode='bicubic')

    # 将插值后的位置编码转换回原始形状
    positional_encoding = positional_encoding.reshape(b, new_num_patches, d)
    positional_encoding = torch.cat((cls_encoding, positional_encoding), dim=1)

    return nn.Parameter(positional_encoding)
