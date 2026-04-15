"""BDC-TransUNet trainer."""

import logging
import os
import sys
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision import transforms

from utils import DiceLoss, calculate_metrics_binary


def create_scheduler(optimizer, args, max_iterations):
    """Create learning rate scheduler."""
    if args.lr_scheduler == 'poly':
        return None

    elif args.lr_scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)

    elif args.lr_scheduler == 'warmup_cosine':
        warmup_iters = args.warmup_epochs * (max_iterations // args.max_epochs)

        def warmup_cosine_lambda(step):
            if step < warmup_iters:
                return float(step) / float(max(1, warmup_iters))
            else:
                progress = float(step - warmup_iters) / float(max(1, max_iterations - warmup_iters))
                return max(args.min_lr / args.base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    else:
        return None


def trainer_medical(args, model, snapshot_path):
    """Training loop for medical image segmentation."""
    from datasets import MedicalDataset, RandomGenerator, ValGenerator

    logger = logging.getLogger()
    logger.handlers = []

    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logging.info(str(args))


    db_train = MedicalDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split='train',
        transform=transforms.Compose([
            RandomGenerator(output_size=[args.img_size, args.img_size])
        ])
    )


    val_path = args.root_path.replace('train_npz', 'val_npz')
    if os.path.exists(val_path):
        db_val = MedicalDataset(
            base_dir=val_path,
            list_dir=args.list_dir,
            split='val',
            transform=transforms.Compose([
                ValGenerator(output_size=[args.img_size, args.img_size])
            ])
        )
    else:
        db_val = None
        logging.warning("Validation set not found: %s", val_path)
        logging.warning("best_model.pth will not be saved automatically.")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    if db_val is not None:
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    else:
        valloader = None

    logging.info(f"Train set size: {len(db_train)}")
    if db_val:
        logging.info(f"Val set size: {len(db_val)}")


    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()


    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    ds_weight = getattr(args, 'ds_weight', 0.4)


    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )


    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))


    max_epoch = args.max_epochs
    accum_steps = args.accumulation_steps
    iters_per_epoch = len(trainloader) // accum_steps
    max_iterations = max_epoch * iters_per_epoch

    scheduler = create_scheduler(optimizer, args, max_iterations)

    logging.info(f"{len(trainloader)} batches per epoch, {iters_per_epoch} updates per epoch (accum={accum_steps})")
    logging.info(f"{max_iterations} total weight updates")


    use_amp = getattr(args, 'use_amp', True)
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logging.info("Using AMP (Automatic Mixed Precision)")


    best_dice = 0.0
    best_epoch = 0
    iter_num = 0

    iterator = tqdm(range(max_epoch), ncols=100, desc="Training")

    for epoch_num in iterator:
        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_ce_loss = 0
        batch_count = 0

        optimizer.zero_grad()

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if use_amp:
                with autocast():
                    model_output = model(image_batch)

                    if isinstance(model_output, tuple):
                        outputs, aux_outputs = model_output
                    else:
                        outputs = model_output
                        aux_outputs = None

                    loss_ce = ce_loss(outputs, label_batch.long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)

                    if args.loss_type == 'dice':
                        loss = loss_dice
                    elif args.loss_type == 'ce':
                        loss = loss_ce
                    elif args.loss_type == 'dice_ce':
                        loss = args.dice_weight * loss_dice + args.ce_weight * loss_ce
                    else:
                        loss = 0.5 * loss_dice + 0.5 * loss_ce


                    if aux_outputs is not None:
                        n_aux = len(aux_outputs)
                        for idx, aux_pred in enumerate(aux_outputs):
                            aux_pred_up = F.interpolate(
                                aux_pred, size=label_batch.shape[1:],
                                mode='bilinear', align_corners=False
                            )
                            aux_ce = ce_loss(aux_pred_up, label_batch.long())
                            aux_dice = dice_loss(aux_pred_up, label_batch, softmax=True)
                            layer_weight = ds_weight * (idx + 1) / n_aux
                            loss = loss + layer_weight * (0.5 * aux_ce + 0.5 * aux_dice)

                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (i_batch + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if scheduler is None:
                        lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_
                    elif args.lr_scheduler == 'warmup_cosine':
                        scheduler.step()
                        lr_ = optimizer.param_groups[0]['lr']
                    else:
                        lr_ = optimizer.param_groups[0]['lr']

                    iter_num += 1
                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss.item() * accum_steps, iter_num)
            else:
                model_output = model(image_batch)

                if isinstance(model_output, tuple):
                    outputs, aux_outputs = model_output
                else:
                    outputs = model_output
                    aux_outputs = None

                loss_ce = ce_loss(outputs, label_batch.long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)

                if args.loss_type == 'dice':
                    loss = loss_dice
                elif args.loss_type == 'ce':
                    loss = loss_ce
                elif args.loss_type == 'dice_ce':
                    loss = args.dice_weight * loss_dice + args.ce_weight * loss_ce
                else:
                    loss = 0.5 * loss_dice + 0.5 * loss_ce


                if aux_outputs is not None:
                    n_aux = len(aux_outputs)
                    for idx, aux_pred in enumerate(aux_outputs):
                        aux_pred_up = F.interpolate(
                            aux_pred, size=label_batch.shape[1:],
                            mode='bilinear', align_corners=False
                        )
                        aux_ce = ce_loss(aux_pred_up, label_batch.long())
                        aux_dice = dice_loss(aux_pred_up, label_batch, softmax=True)
                        layer_weight = ds_weight * (idx + 1) / n_aux
                        loss = loss + layer_weight * (0.5 * aux_ce + 0.5 * aux_dice)

                loss = loss / accum_steps
                loss.backward()

                if (i_batch + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is None:
                        lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_
                    elif args.lr_scheduler == 'warmup_cosine':
                        scheduler.step()
                        lr_ = optimizer.param_groups[0]['lr']
                    else:
                        lr_ = optimizer.param_groups[0]['lr']

                    iter_num += 1
                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss.item() * accum_steps, iter_num)

            epoch_loss += loss.item() * accum_steps
            epoch_dice_loss += loss_dice.item()
            epoch_ce_loss += loss_ce.item()
            batch_count += 1


        if scheduler is not None and args.lr_scheduler == 'cosine':
            scheduler.step()


        avg_loss = epoch_loss / batch_count
        avg_dice = epoch_dice_loss / batch_count
        avg_ce = epoch_ce_loss / batch_count

        logging.info(f'Epoch [{epoch_num}/{max_epoch}] Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, CE: {avg_ce:.4f}')
        writer.add_scalar('epoch/loss', avg_loss, epoch_num)
        writer.add_scalar('epoch/dice_loss', avg_dice, epoch_num)
        writer.add_scalar('epoch/ce_loss', avg_ce, epoch_num)


        if valloader is not None:
            model.eval()
            val_dice, val_iou = validate(model, valloader, args)
            model.train()

            logging.info(f'  Validation: Dice={val_dice:.4f}, IoU={val_iou:.4f}')
            writer.add_scalar('val/dice', val_dice, epoch_num)
            writer.add_scalar('val/iou', val_iou, epoch_num)

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch_num
                best_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                logging.info(f'  New best model! Dice={best_dice:.4f} (Epoch {best_epoch})')


        if (epoch_num + 1) % 50 == 0:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_path)


    final_path = os.path.join(snapshot_path, 'final_model.pth')
    torch.save(model.state_dict(), final_path)

    writer.close()
    logging.info(f"Training finished! Best Dice: {best_dice:.4f} at Epoch {best_epoch}")

    return best_dice


def validate(model, valloader, args):
    """Run validation and return mean Dice and IoU."""
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for sampled_batch in valloader:
            image, label = sampled_batch['image'], sampled_batch['label']
            image, label = image.cuda(), label.cuda()

            outputs = model(image)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)

            pred = outputs.cpu().numpy().flatten()
            gt = label.cpu().numpy().flatten()

            result = calculate_metrics_binary(pred, gt)
            dice, iou = result[0], result[1]
            dice_scores.append(dice)
            iou_scores.append(iou)

    return np.mean(dice_scores), np.mean(iou_scores)
