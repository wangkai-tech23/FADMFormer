import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from colorama import Fore, Style
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset import TestSetLoader, TrainSetLoader
from FADMFormer import FADMFormer
from metric import PD_FA, ROCMetric, SamplewiseSigmoidMetric, SigmoidMetric


DATASET_DEFAULTS = {
    "NUDT-SIRST": {"size": 256, "batch": 16, "split_method": "", "aug": 0.0},
    "IRSTD-1k": {"size": 512, "batch": 4, "split_method": "", "aug": 0.0},
    "NUAA-SIRST": {"size": 512, "batch": 4, "split_method": "", "aug": 0.0},
}


def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


class SoftIoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = (target.float() > 0.5).float()
        dims = tuple(range(1, pred.dim()))
        inter = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
        iou = (inter + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train FADMFormer for infrared small target detection")

    # Dataset
    parser.add_argument("--dataset", type=str, default="IRSTD-1k", choices=["IRSTD-1k", "NUDT-SIRST", "NUAA-SIRST"])
    parser.add_argument("--root", type=str, default="datasets")
    parser.add_argument("--split_method", type=str, default=None,
                        help="Optional split folder. If empty, use dataset root directly.")
    parser.add_argument("--train_meta", type=str, default="",
                        help="Path to train txt. If empty, auto-search train.txt.")
    parser.add_argument("--val_meta", type=str, default="",
                        help="Path to val/test txt. If empty, auto-search val_test.txt/test.txt.")
    parser.add_argument("--form", type=str, default=".png")
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--aug", type=float, default=None)
    parser.add_argument("--workers", type=int)

    # Model
    parser.add_argument("--input_channels", type=int)
    parser.add_argument("--nb_filter", type=int, nargs="+")
    parser.add_argument("--depth", type=int, nargs="+")
    parser.add_argument("--win_size", type=int)
    parser.add_argument("--drop", type=float)
    parser.add_argument("--attn_drop", type=float)
    parser.add_argument("--drop_path", type=float)

    # Optimization, following the paper setting by default
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--optimizer", type=str, default="Adagrad", choices=["Adagrad", "AdamW", "Adam", "SGD"])
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--tmax", type=int)
    parser.add_argument("--seed", type=int)

    # Resume/save
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_root", type=str, default="runs")
    parser.add_argument("--save_after", type=int, default=100)
    parser.add_argument("--config", type=str, default="options/traintestcfg.yml",
                        help="Optional yaml config. CLI args override config if provided.")

    args = parser.parse_args()
    return args


def read_ids(txt_path: str) -> List[str]:
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def resolve_dataset_paths(args) -> Tuple[str, str, str, int, int, float, str]:
    cfg = {}
    cfg_path = Path(args.config)
    if args.config and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader) or {}

    dataset_cfg = DATASET_DEFAULTS.get(args.dataset, {}).copy()
    if cfg.get("dataset"):
        root_from_cfg = cfg["dataset"].get("root", None)
        if root_from_cfg and args.root == "datasets":
            args.root = root_from_cfg
        if args.dataset in cfg["dataset"]:
            dataset_cfg.update(cfg["dataset"][args.dataset])

    split_method = args.split_method
    if split_method is None:
        split_method = str(dataset_cfg.get("split_method", "") or "")

    img_size = int(args.img_size if args.img_size > 0 else dataset_cfg.get("size", 512))
    batch_size = int(args.batch_size if args.batch_size > 0 else dataset_cfg.get("batch", 4))
    aug = float(args.aug if args.aug is not None else dataset_cfg.get("aug", 0.0))

    dataset_dir = os.path.join(args.root, args.dataset)
    split_dir = os.path.join(dataset_dir, split_method) if split_method else dataset_dir

    train_meta = args.train_meta
    val_meta = args.val_meta
    if not train_meta:
        candidates = [
            os.path.join(split_dir, "train.txt"),
            os.path.join(dataset_dir, "train.txt"),
        ]
        train_meta = next((p for p in candidates if os.path.isfile(p)), candidates[0])
    if not val_meta:
        candidates = [
            os.path.join(split_dir, "val_test.txt"),
            os.path.join(split_dir, "test.txt"),
            os.path.join(dataset_dir, "val_test.txt"),
            os.path.join(dataset_dir, "test.txt"),
        ]
        val_meta = next((p for p in candidates if os.path.isfile(p)), candidates[0])

    if not os.path.isfile(train_meta):
        raise FileNotFoundError(f"train_meta not found: {train_meta}")
    if not os.path.isfile(val_meta):
        raise FileNotFoundError(f"val_meta not found: {val_meta}")

    return dataset_dir, train_meta, val_meta, img_size, batch_size, aug, split_method


def build_model(args, img_size: int, device: torch.device):
    model = FADMFormer(
        input_channels=args.input_channels,
        nb_filter=args.nb_filter,
        depth=args.depth,
        drop=args.drop,
        attn_drop=args.attn_drop,
        drop_path=args.drop_path,
        num_heads=[[1, 2, 3, 4], [1, 3, 4], [1, 3], [1]],
        win_size=args.win_size,
        img_size=img_size,
    )
    return model.to(device)


def build_optimizer(args, model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = args.optimizer.lower()
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    if args.scheduler.lower() == "cosineannealinglr":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.min_lr)
    if args.scheduler.lower() in ("none", ""):
        return None
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device=None):
    ckpt = torch.load(path, map_location=device or "cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    new_state = {}
    for k, v in state.items():
        new_state[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(new_state, strict=True)
    start_epoch = int(ckpt.get("epoch", 0)) + 1 if isinstance(ckpt, dict) else 1
    best_iou = float(ckpt.get("IOU", ckpt.get("mean_IOU", 0.0))) if isinstance(ckpt, dict) else 0.0
    best_niou = float(ckpt.get("n_IoU", 0.0)) if isinstance(ckpt, dict) else 0.0
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and isinstance(ckpt, dict) and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return start_epoch, best_iou, best_niou


def append_csv(path: str, row: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    clean = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            clean[k] = json.dumps(v.tolist())
        elif isinstance(v, np.generic):
            clean[k] = float(v)
        else:
            clean[k] = v
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(clean)


def train_one_epoch(model, optimizer, data_loader, device, epoch: int, loss_fn):
    model.train()
    losses = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(data_loader, file=sys.stdout, ncols=120,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Style.RESET_ALL))
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = (labels > 0).float().to(device, non_blocking=True)

        pred = model(images)
        if isinstance(pred, (list, tuple)):
            loss = sum(loss_fn(p, labels) for p in pred) / len(pred)
            pred_use = pred[-1]
        else:
            loss = loss_fn(pred, labels)
            pred_use = pred

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.update(loss.item(), images.size(0))
        pbar.set_description(f"[train epoch {epoch:04d}] loss: {losses.avg:.8f}")
    return losses.avg


@torch.no_grad()
def evaluate(model, data_loader, device, epoch: int, loss_fn, bins: int = 10):
    model.eval()
    iou_metric = SigmoidMetric(score_thresh=0.5)
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    pdfa = PD_FA(1, bins)
    roc = ROCMetric(1, bins)
    losses = AverageMeter()

    pbar = tqdm(data_loader, file=sys.stdout, ncols=120,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Style.RESET_ALL))
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = (labels > 0).float().to(device, non_blocking=True)

        pred = model(images)
        if isinstance(pred, (list, tuple)):
            loss = sum(loss_fn(p, labels) for p in pred) / len(pred)
            pred = pred[-1]
        else:
            loss = loss_fn(pred, labels)

        losses.update(loss.item(), images.size(0))
        pred_cpu = pred.detach().cpu()
        labels_cpu = labels.detach().cpu()
        iou_metric.update(pred_cpu, labels_cpu)
        niou_metric.update(pred_cpu, labels_cpu)
        roc.update(pred_cpu, labels_cpu)
        pdfa.update(pred_cpu, labels_cpu)

        _, iou = iou_metric.get()
        _, niou = niou_metric.get()
        pbar.set_description(f"[valid epoch {epoch:04d}] loss: {losses.avg:.6f}, IoU: {iou:.6f}, nIoU: {niou:.6f}")

    _, iou = iou_metric.get()
    _, niou = niou_metric.get()
    tpr, fpr, recall, precision = roc.get()
    fa, pd = pdfa.get()
    return losses.avg, iou, niou, tpr, fpr, recall, precision, pd, fa


def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss, iou, niou):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": val_loss,
            "IOU": iou,
            "n_IoU": niou,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_dir, train_meta, val_meta, img_size, batch_size, aug, split_method = resolve_dataset_paths(args)

    train_img_ids = read_ids(train_meta)
    val_img_ids = read_ids(val_meta)
    print(f"dataset_dir: {dataset_dir}")
    print(f"train_meta : {train_meta} ({len(train_img_ids)} samples)")
    print(f"val_meta   : {val_meta} ({len(val_img_ids)} samples)")
    print(f"img_size={img_size}, train_batch={batch_size}, val_batch=1, aug={aug}")

    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    trainset = TrainSetLoader(
        dataset_dir,
        img_id=train_img_ids,
        base_size=img_size,
        crop_size=img_size,
        transform=input_transform,
        form=args.form,
        aug=aug,
        useprior=True,
    )
    valset = TestSetLoader(
        dataset_dir,
        img_id=val_img_ids,
        base_size=img_size,
        crop_size=img_size,
        transform=input_transform,
        form=args.form,
    )

    train_data = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    # Use batch size 1 for object-level Pd/Fa consistency and easier debugging.
    val_data = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )

    model = build_model(args, img_size, device)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"# model parameters: {params_m:.2f} M")

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    loss_fn = SoftIoULoss().to(device)

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    split_tag = split_method if split_method else "default"
    log_dir = os.path.join(args.save_root, f"{current_time}_{args.dataset}_{split_tag}_FADMFormer")
    ckpt_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir)
    metrics_csv = os.path.join(log_dir, "metrics_history.csv")

    start_epoch = 1
    best_iou = 0.0
    best_niou = 0.0
    best_iou_name = ""
    best_niou_name = ""
    if args.resume:
        start_epoch, best_iou, best_niou = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_iou={best_iou:.4f}, best_nIoU={best_niou:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_data, device, epoch, loss_fn)
        val_loss, iou, niou, tpr, fpr, recall, precision, pd, fa = evaluate(model, val_data, device, epoch, loss_fn, bins=10)
        if scheduler is not None:
            scheduler.step()

        idx05 = 5
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "lr": lr,
            "time_sec": elapsed,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "IoU": iou,
            "nIoU": niou,
            "PD@0.5": float(pd[idx05]),
            "FA@0.5": float(fa[idx05]),
        }
        append_csv(metrics_csv, record)

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("IoU", iou, epoch)
        tb_writer.add_scalar("nIoU", niou, epoch)
        tb_writer.add_scalar("PD@0.5", float(pd[idx05]), epoch)
        tb_writer.add_scalar("FA@0.5", float(fa[idx05]), epoch)
        tb_writer.add_scalar("lr", lr, epoch)

        print(
            f"[Epoch {epoch:04d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"IoU={iou:.6f} nIoU={niou:.6f} PD@0.5={pd[idx05]:.6f} FA@0.5={fa[idx05]:.8f} "
            f"lr={lr:.6g} time={elapsed:.1f}s"
        )

        latest_path = os.path.join(ckpt_dir, "latest.pth.tar")
        save_checkpoint(latest_path, epoch, model, optimizer, scheduler, val_loss, iou, niou)

        if epoch >= args.save_after:
            name = f"Epoch-{epoch:04d}_IoU-{iou:.4f}_nIoU-{niou:.4f}.pth.tar"
            if iou > best_iou:
                best_path = os.path.join(ckpt_dir, "Best_IoU_" + name)
                save_checkpoint(best_path, epoch, model, optimizer, scheduler, val_loss, iou, niou)
                if best_iou_name and os.path.exists(os.path.join(ckpt_dir, "Best_IoU_" + best_iou_name)):
                    os.remove(os.path.join(ckpt_dir, "Best_IoU_" + best_iou_name))
                best_iou = iou
                best_iou_name = name
            if niou > best_niou:
                best_path = os.path.join(ckpt_dir, "Best_nIoU_" + name)
                save_checkpoint(best_path, epoch, model, optimizer, scheduler, val_loss, iou, niou)
                if best_niou_name and os.path.exists(os.path.join(ckpt_dir, "Best_nIoU_" + best_niou_name)):
                    os.remove(os.path.join(ckpt_dir, "Best_nIoU_" + best_niou_name))
                best_niou = niou
                best_niou_name = name

    tb_writer.close()


if __name__ == "__main__":
    main(parse_args())
