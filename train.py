"""BDC-TransUNet training script."""

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks import BDCTransUNet, CONFIGS


def get_args():
    parser = argparse.ArgumentParser(description='BDC-TransUNet Training')

    # Dataset
    parser.add_argument('--dataset', type=str, default='GLAS',
                        choices=['GLAS', 'Kvasir', 'CVC'],
                        help='Dataset name')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Root path of dataset (auto-set if None)')
    parser.add_argument('--list_dir', type=str, default=None,
                        help='List directory (auto-set if None)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for checkpoints')
    parser.add_argument('--exp', type=str, default='BDC_experiment',
                        help='Experiment name')

    # Model architecture
    parser.add_argument('--encoder_type', type=str, default='bskan',
                        choices=['mlp', 'kan', 'bskan'],
                        help='Encoder MLP type')
    parser.add_argument('--attention_type', type=str, default='dsda',
                        choices=['none', 'se', 'dsda'],
                        help='Decoder attention type')
    parser.add_argument('--upsample_type', type=str, default='converse',
                        choices=['bilinear', 'converse', 'deconv'],
                        help='Decoder upsampling type')

    # BSKAN parameters
    parser.add_argument('--grid_size', type=int, default=3,
                        help='KAN grid size')
    parser.add_argument('--spline_order', type=int, default=2,
                        help='KAN spline order')
    parser.add_argument('--boundary_threshold', type=float, default=0.3,
                        help='BSKAN boundary threshold')

    # Decoder parameters
    parser.add_argument('--da_reduction', type=int, default=16,
                        help='DSDA/SE channel reduction ratio')
    parser.add_argument('--enhance_layers', type=str, default='0,1,2',
                        help='Decoder layers to apply attention (comma separated)')

    # Deep supervision
    parser.add_argument('--ds_weight', type=float, default=0.4,
                        help='Deep supervision loss weight')

    # Training
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size per GPU')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'cosine', 'warmup_cosine', 'step'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs for warmup_cosine scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # Model structure
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='Number of skip connections')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='ViT model name')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                        help='ViT patch size')

    # Loss
    parser.add_argument('--loss_type', type=str, default='dice_ce',
                        choices=['dice', 'ce', 'dice_ce', 'focal', 'boundary'],
                        help='Loss function type')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for dice loss in dice_ce')
    parser.add_argument('--ce_weight', type=float, default=0.5,
                        help='Weight for CE loss in dice_ce')

    # Misc
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='Whether to use deterministic training')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--pretrained_path', type=str,
                        default='./pretrained_models/R50+ViT-B_16.npz',
                        help='Path to pretrained ViT weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')
    parser.add_argument('--use_amp', type=int, default=0,
                        help='Use automatic mixed precision (1=True, 0=False)')

    args = parser.parse_args()
    return args


def setup_dataset_config(args):
    """Auto-configure dataset paths."""
    dataset_config = {
        'GLAS': {
            'root_path': './data/GLAS/train_npz',
            'list_dir': './lists/lists_GLAS',
            'num_classes': 2,
        },
        'Kvasir': {
            'root_path': './data/Kvasir/train_npz',
            'list_dir': './lists/lists_Kvasir',
            'num_classes': 2,
        },
        'CVC': {
            'root_path': './data/CVC/train_npz',
            'list_dir': './lists/lists_CVC',
            'num_classes': 2,
        },
    }

    config = dataset_config[args.dataset]
    if args.root_path is None:
        args.root_path = config['root_path']
    if args.list_dir is None:
        args.list_dir = config['list_dir']
    args.num_classes = config['num_classes']

    return args


def create_model(args):
    """Create model with specified configuration."""
    config = CONFIGS[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip

    if args.vit_name.find('R50') != -1:
        config.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )

    # Parse enhance_layers
    if args.enhance_layers and args.enhance_layers.lower() != 'none':
        enhance_layers = [int(x) for x in args.enhance_layers.split(',')]
    else:
        enhance_layers = None

    model = BDCTransUNet(
        config=config,
        img_size=args.img_size,
        num_classes=args.num_classes,
        encoder_type=args.encoder_type,
        attention_type=args.attention_type,
        upsample_type=args.upsample_type,
        enhance_layers=enhance_layers,
        da_reduction=args.da_reduction,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        boundary_threshold=args.boundary_threshold,
        vis=False
    )

    # Load pretrained weights
    if args.pretrained_path:
        if not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found: {args.pretrained_path}")
        print(f"Loading pretrained weights from {args.pretrained_path}")
        weights = np.load(args.pretrained_path)
        model.load_from(weights)

    return model


def main():
    args = get_args()
    args = setup_dataset_config(args)

    # Set random seed
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create output directory
    snapshot_path = os.path.join(args.output_dir, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # Logging
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # Print configuration
    print("=" * 60)
    print("BDC-TransUNet Training")
    print("=" * 60)
    print(f"Dataset:      {args.dataset}")
    print(f"Encoder:      {args.encoder_type}")
    print(f"Attention:    {args.attention_type}")
    print(f"Upsample:     {args.upsample_type}")
    print(f"Seed:         {args.seed}")
    print(f"Epochs:       {args.max_epochs}")
    print(f"Batch Size:   {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}")
    print(f"Learning Rate:{args.base_lr}")
    print("=" * 60)

    logging.info(str(args))

    # Create model
    model = create_model(args)
    model = model.cuda()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params / 1e6:.4f}M")
    print(f"Trainable params: {trainable_params / 1e6:.4f}M")
    logging.info(f"Total params: {total_params / 1e6:.4f}M")

    # Train
    from trainer import trainer_medical
    trainer_medical(args, model, snapshot_path)


if __name__ == '__main__':
    main()
