"""BDC-TransUNet testing script."""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_medical import MedicalDataset
from utils.metrics import calculate_metrics_comprehensive
from networks import BDCTransUNet, CONFIGS


def set_deterministic(seed=42):
    """Set deterministic computation for reproducible testing."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='BDC-TransUNet Testing')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['GLAS', 'Kvasir', 'CVC'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override default data directory')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    # Model
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--encoder_type', type=str, default='bskan',
                        choices=['mlp', 'kan', 'bskan'],
                        help='Encoder MLP type')
    parser.add_argument('--attention_type', type=str, default='dsda',
                        choices=['none', 'se', 'dsda'],
                        help='Decoder attention type')
    parser.add_argument('--upsample_type', type=str, default='converse',
                        choices=['bilinear', 'converse', 'deconv'],
                        help='Decoder upsampling type')
    parser.add_argument('--grid_size', type=int, default=3, help='KAN grid size')
    parser.add_argument('--spline_order', type=int, default=2, help='KAN spline order')
    parser.add_argument('--boundary_threshold', type=float, default=0.3, help='BSKAN boundary threshold')
    parser.add_argument('--da_reduction', type=int, default=16, help='DSDA/SE channel reduction ratio')
    parser.add_argument('--enhance_layers', type=str, default='0,1,2',
                        help='Decoder layers with attention (comma separated)')
    parser.add_argument('--n_skip', type=int, default=3, help='Number of skip connections')

    # Output
    parser.add_argument('--output_file', type=str, default=None, help='Output file path')
    parser.add_argument('--test_seed', type=int, default=42, help='Random seed for testing')

    return parser.parse_args()


DATASET_CONFIG = {
    'GLAS': {
        'root_path': './data/GLAS/test_npz',
        'list_dir': './lists/lists_GLAS',
    },
    'Kvasir': {
        'root_path': './data/Kvasir/test_npz',
        'list_dir': './lists/lists_Kvasir',
    },
    'CVC': {
        'root_path': './data/CVC/test_npz',
        'list_dir': './lists/lists_CVC',
    },
}


def create_model(args):
    """Create model with specified configuration."""
    config = CONFIGS['R50-ViT-B_16']
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    config.patches.grid = (args.img_size // 16, args.img_size // 16)

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
    )

    return model


def test_model(args):
    """Evaluate model on test set."""

    set_deterministic(args.test_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nModel configuration:")
    print(f"  Encoder:   {args.encoder_type}")
    print(f"  Attention: {args.attention_type}")
    print(f"  Upsample:  {args.upsample_type}")

    model = create_model(args)

    # Load checkpoint
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return None

    print(f"\nLoading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load test data
    dataset_config = DATASET_CONFIG[args.dataset]
    data_dir = args.data_dir if args.data_dir else dataset_config['root_path']

    test_dataset = MedicalDataset(
        base_dir=data_dir,
        list_dir=dataset_config['list_dir'],
        split='test_vol'
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nDataset: {args.dataset}")
    print(f"Test samples: {len(test_dataset)}")

    all_metrics = defaultdict(list)

    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            image = batch['image']
            label = batch['label'].numpy()
            original_size = (label.shape[-2], label.shape[-1])

            if image.shape[-1] != args.img_size or image.shape[-2] != args.img_size:
                image = torch.nn.functional.interpolate(
                    image, size=(args.img_size, args.img_size),
                    mode='bilinear', align_corners=False
                )

            image = image.to(device)

            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, list):
                output = output[-1]

            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

            if pred.shape[-1] != original_size[1] or pred.shape[-2] != original_size[0]:
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1).float(),
                    size=original_size,
                    mode='nearest'
                ).squeeze(1).long()

            pred = pred.cpu().numpy()

            for i in range(pred.shape[0]):
                label_i = label[i] if label.ndim > 2 else label
                label_i = np.squeeze(label_i)
                pred_i = pred[i] if pred.ndim > 2 else pred
                pred_i = np.squeeze(pred_i)

                metrics = calculate_metrics_comprehensive(pred_i, label_i)
                for key, value in metrics.items():
                    all_metrics[key].append(value)

    # Print results
    results = {}
    print("\n" + "=" * 70)
    print(f"Results - {args.dataset}")
    print("=" * 70)
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)

    metric_order = [
        'Dice', 'IoU', 'HD95', 'HD', 'ASSD',
        'Precision', 'Recall', 'Specificity', 'Accuracy',
        'F2', 'MCC', 'VS'
    ]

    for metric_name in metric_order:
        if metric_name in all_metrics:
            values = np.array(all_metrics[metric_name])
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)

                results[metric_name] = {
                    'mean': mean_val, 'std': std_val,
                    'min': min_val, 'max': max_val
                }

                if metric_name in ['HD95', 'HD', 'ASSD']:
                    print(f"{metric_name:<15} {mean_val:<12.2f} {std_val:<12.2f} {min_val:<12.2f} {max_val:<12.2f}")
                else:
                    print(f"{metric_name:<15} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")

    print("=" * 70)

    # Save results
    if args.output_file:
        output_path = args.output_file
    else:
        output_dir = os.path.dirname(args.model_path)
        output_path = os.path.join(output_dir, f'test_results_{args.dataset}.txt')

    output_dir_path = os.path.dirname(output_path)
    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"BDC-TransUNet Test Results - {args.dataset}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Config: encoder={args.encoder_type}, attention={args.attention_type}, upsample={args.upsample_type}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 70 + "\n")

        for metric_name in metric_order:
            if metric_name in results:
                r = results[metric_name]
                if metric_name in ['HD95', 'HD', 'ASSD']:
                    f.write(f"{metric_name:<15} {r['mean']:<12.2f} {r['std']:<12.2f} {r['min']:<12.2f} {r['max']:<12.2f}\n")
                else:
                    f.write(f"{metric_name:<15} {r['mean']:<12.4f} {r['std']:<12.4f} {r['min']:<12.4f} {r['max']:<12.4f}\n")

        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    args = get_args()
    test_model(args)
