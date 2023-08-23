import argparse
import json
from fnmatch import fnmatch

from einops.layers.torch import Rearrange
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models.vision_transformer import Block
import torch
from pytorch_pretrained_vit import ViT
from torch import nn
from torchinfo import summary

from Utils.LoadClevr import LoadClevr
from Utils.Seed_everything import Seed_everything
from Utils.Train_Classification import Train_Classification
from Utils.Utils import SmoothCrossEntropyLoss, get_params_groups, cosine_scheduler


class Ensemble(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, num_classes):
        super(Ensemble, self).__init__()
        layer = 24
        self.model = ViT('B_16', pretrained=False, dropout_rate=args.dropout)
        self.model.load_state_dict(torch.load(r'..\ckpt\B_16.pth'), strict=False)

        for name, param in self.model.named_parameters():
            if name != 'transformer.promote':
                param.requires_grad = False
            else:
                print('promote activate')

        for _, (name, param) in enumerate(self.model.named_modules()):
            print(name)
            if fnmatch(name, 'transformer.blocks.*.norm2') or fnmatch(name, 'transformer.blocks.*.norm1'):
                param.register_forward_hook(self.get_feature1)

        self.proj_model1 = nn.Sequential(
            Rearrange('layer batch token dim -> batch (token dim) layer'),
            nn.Linear(layer, layer, bias=False),
            nn.GELU(),
            nn.Linear(layer, layer, bias=False),
            Rearrange('batch (token dim) layer -> batch token (layer dim)', token=(args.image_size//args.patch_size)**2+1, dim=embed_dim, layer=layer),

            # Rearrange('layer batch token dim -> batch token (layer dim)'),
            nn.LayerNorm(embed_dim * layer),
            nn.Linear(embed_dim * layer, embed_dim),
            nn.Dropout(),
            *[Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(1)],
            nn.LayerNorm(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

        # ----------------------------------------------------------
        # self.beta = args.beta
        # self.ema = nn.Sequential(
        #     Rearrange('layer batch token dim -> batch (token dim) layer'),
        #     nn.Linear(layer, layer, bias=False),
        #     nn.GELU(),
        #     nn.Linear(layer, layer, bias=False),
        #     Rearrange('batch (token dim) layer -> batch token (layer dim)', token=(args.image_size//args.patch_size)**2+11, dim=embed_dim, layer=layer),
        #
        #     nn.LayerNorm(embed_dim * layer),
        #     nn.Linear(embed_dim * layer, embed_dim),
        #     *[Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(1)],
        #     nn.LayerNorm(embed_dim)
        # )
        # self.classifierEma = nn.Linear(embed_dim, num_classes)
        # self.classifierEma.weight.data.normal_(mean=0.0, std=0.01)
        # self.classifierEma.bias.data.zero_()
        # for param_shadow in self.ema.parameters():
        #     param_shadow.requires_grad = False
        # for param_shadow in self.classifierEma.parameters():
        #     param_shadow.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        for param_shadow, param_online in zip(self.ema.parameters(), self.proj_model1.parameters()):
            param_shadow.data = param_shadow.data * self.beta + param_online.data * (1. - self.beta)
        for param_shadow, param_online in zip(self.classifierEma.parameters(), self.classifier.parameters()):
            param_shadow.data = param_shadow.data * self.beta + param_online.data * (1. - self.beta)

    def get_feature1(self, module, data_input, data_output):
        self.features1.append(data_output)

    def forward(self, x):
        self.features1 = []
        _ = self.model(x)

        self.features1 = torch.stack(self.features1)
        self.features1 = self.proj_model1(self.features1).mean(dim=1)
        out = self.classifier(self.features1)
        return out

    def forward_ema(self, x):
        with torch.no_grad():
            self.features1 = []
            _ = self.model(x)

            self.features1 = torch.stack(self.features1)
            self.features1 = self.ema(self.features1).mean(dim=1)
            out = self.classifierEma(self.features1)
            return out


parser = argparse.ArgumentParser()
parser.add_argument('--EXP_NUM', default='{ViT-Clevr-Fuse-1e3-10epoch}', type=str)
parser.add_argument('--dataset_path', default=r'..\data', type=str)
parser.add_argument("--patch_size", default=16, type=int)
parser.add_argument("--image_size", default=224, type=int)
parser.add_argument("--num_heads", default=12, type=int)
parser.add_argument("--mlp_ratio", default=4., type=int)
parser.add_argument("--num_classes", default=11, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument("--warmup_epochs", default=1, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument('--min_lr', default=0., type=float)
parser.add_argument('--weight_decay', default=1e-1, type=float)
parser.add_argument('--weight_decay_end', default=1e-1, type=float)
parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
parser.add_argument('--dropout', default=0., type=float)
parser.add_argument('--drop_path', default=0., type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--if_ema", default=False, type=bool)
parser.add_argument("--beta", default=0.999, type=float)
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--root_path", default='./', type=str)
parser.add_argument('--epsilon', default=1e-8, type=float)
# * Random Erase params
parser.add_argument('--reprob', type=float, default=0., metavar='PCT', help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
# * Mixup params
parser.add_argument('--mixup', type=float, default=0., help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0., help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=0.,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
args = parser.parse_args()
vit_base = {'emb_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4., 'name': 'vit_base'}
vit_large = {'emb_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4., 'name': 'vit_large'}
model_config = vit_base
if __name__ == '__main__':
    Seed_everything(seed_value=args.seed)
    model = Ensemble(embed_dim=model_config['emb_dim'], num_classes=args.num_classes, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio)
    summary(model, input_size=(1, 3, args.image_size, args.image_size), depth=3)

    train_loader, valid_loader = LoadClevr(root_dir=args.dataset_path, batch_size=args.batch_size, image_size=args.image_size)

    params_groups = get_params_groups(model)
    lr_schedule = cosine_scheduler(
        args.lr * args.batch_size / 256.,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
        final_value=args.min_lr,
    )
    optimizer = torch.optim.AdamW(params_groups, betas=args.betas, eps=args.epsilon)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = SmoothCrossEntropyLoss(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    args_run = json.dumps(vars(args))
    root = args.root_path
    with open(root + str(args.EXP_NUM) + "_config.txt", 'w') as f:
        f.write(args_run)
    runs = root + str(args.EXP_NUM)
    Train_Classification(model, train_loader, valid_loader, args.epochs, args.num_classes, criterion=criterion,
                         optimizer=optimizer, device=args.device, recoverFromNEpoch=0,
                         lr_schedule=lr_schedule, class_distribute=None, if_ema=args.if_ema,
                         mixup_fn=mixup_fn, EXP=args.EXP_NUM)
