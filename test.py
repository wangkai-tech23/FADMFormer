import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import TestSetLoader
from FADMFormer import FADMFormer
from metric import PD_FA, ROCMetric, SamplewiseSigmoidMetric, SigmoidMetric


DATASET_DEFAULTS = {
    "NUDT-SIRST": {"size": 256, "batch": 1, "split_method": ""},
    "IRSTD-1k": {"size": 512, "batch": 1, "split_method": ""},
    "NUAA-SIRST": {"size": 512, "batch": 1, "split_method": ""},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test FADMFormer")
    parser.add_argument("--dataset", type=str, default="IRSTD-1k", choices=["IRSTD-1k", "NUDT-SIRST", "NUAA-SIRST"])
    parser.add_argument("--root", type=str, default="datasets")
    parser.add_argument("--split_method", type=str, default=None)
    parser.add_argument("--val_meta", type=str, default="")
    parser.add_argument("--form", type=str, default=".png")
    parser.add_argument("--img_size", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./test_results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--nb_filter", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--depth", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--win_size", type=int, default=8)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.1)
    return parser.parse_args()


def read_ids(txt_path: str) -> List[str]:
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def resolve_dataset_paths(args):
    default = DATASET_DEFAULTS.get(args.dataset, {})
    split_method = args.split_method
    if split_method is None:
        split_method = str(default.get("split_method", "") or "")
    img_size = int(args.img_size if args.img_size > 0 else default.get("size", 512))

    dataset_dir = os.path.join(args.root, args.dataset)
    split_dir = os.path.join(dataset_dir, split_method) if split_method else dataset_dir

    val_meta = args.val_meta
    if not val_meta:
        candidates = [
            os.path.join(split_dir, "val_test.txt"),
            os.path.join(split_dir, "test.txt"),
            os.path.join(dataset_dir, "val_test.txt"),
            os.path.join(dataset_dir, "test.txt"),
        ]
        val_meta = next((p for p in candidates if os.path.isfile(p)), candidates[0])
    if not os.path.isfile(val_meta):
        raise FileNotFoundError(f"val_meta not found: {val_meta}")
    return dataset_dir, val_meta, img_size, split_method


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


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        new_state[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(new_state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict):
        print(f"epoch={ckpt.get('epoch', 'NA')}, IoU={ckpt.get('IOU', 'NA')}, nIoU={ckpt.get('n_IoU', 'NA')}")


def save_mask(path: str, arr: np.ndarray):
    arr = (arr > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def image_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    # img_tensor is normalized RGB [3,H,W]. Unnormalize for visualization.
    x = img_tensor.detach().cpu().float().numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = x * std + mean
    x = np.clip(x, 0.0, 1.0)
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return x


def make_panel(raw_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, prob: np.ndarray, title: str, iou: float) -> Image.Image:
    h, w = gt.shape
    raw = Image.fromarray(raw_rgb).resize((w, h), Image.Resampling.BICUBIC)
    gt_img = Image.fromarray((gt > 0.5).astype(np.uint8) * 255).convert("RGB")
    pred_img = Image.fromarray((pred > 0.5).astype(np.uint8) * 255).convert("RGB")
    prob_img = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).convert("RGB")

    pad = 8
    top = 32
    canvas = Image.new("RGB", (w * 4 + pad * 3, h + top), "white")
    draw = ImageDraw.Draw(canvas)
    names = ["Image", "GT", "Pred", f"Prob IoU={iou:.4f}"]
    imgs = [raw, gt_img, pred_img, prob_img]
    for i, im in enumerate(imgs):
        x0 = i * (w + pad)
        canvas.paste(im, (x0, top))
        draw.text((x0 + 4, 8), names[i], fill=(0, 0, 0))
    return canvas


def sample_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = np.logical_and(pred_bin > 0.5, gt_bin > 0.5).sum()
    union = np.logical_or(pred_bin > 0.5, gt_bin > 0.5).sum()
    return float((inter + 1e-6) / (union + 1e-6))


def append_csv(path: str, row: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    clean = {k: (json.dumps(v.tolist()) if isinstance(v, np.ndarray) else v) for k, v in row.items()}
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(clean)


@torch.no_grad()
def test(model, loader, device, args, ids):
    model.eval()
    iou_metric = SigmoidMetric(score_thresh=args.threshold)
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=args.threshold)
    pdfa = PD_FA(1, 10)
    roc = ROCMetric(1, 10)

    pred_dir = os.path.join(args.save_dir, "pred")
    gt_dir = os.path.join(args.save_dir, "gt")
    vis_dir = os.path.join(args.save_dir, "visual")
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_pred or args.save_vis:
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    details_csv = os.path.join(args.save_dir, "test_details.csv")
    pbar = tqdm(loader, ncols=120, desc="Test")
    global_idx = 0
    for batch in pbar:
        if len(batch) == 3:
            images, labels, batch_ids = batch
        else:
            images, labels = batch
            batch_ids = ids[global_idx:global_idx + images.size(0)]

        images = images.to(device, non_blocking=True)
        labels = (labels > 0).float().to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        probs = torch.sigmoid(logits)
        preds = (probs > args.threshold).float()

        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.detach().cpu()
        iou_metric.update(logits_cpu, labels_cpu)
        niou_metric.update(logits_cpu, labels_cpu)
        roc.update(logits_cpu, labels_cpu)
        pdfa.update(logits_cpu, labels_cpu)

        _, iou = iou_metric.get()
        _, niou = niou_metric.get()
        pbar.set_postfix({"IoU": f"{iou:.4f}", "nIoU": f"{niou:.4f}"})

        b = images.size(0)
        for bi in range(b):
            img_id = str(batch_ids[bi])
            prob_np = probs[bi, 0].detach().cpu().numpy()
            pred_np = preds[bi, 0].detach().cpu().numpy()
            gt_np = labels[bi, 0].detach().cpu().numpy()
            iou_i = sample_iou(pred_np, gt_np)
            append_csv(details_csv, {"id": img_id, "IoU": iou_i, "pred_pixels": float(pred_np.sum()), "gt_pixels": float(gt_np.sum())})
            if args.save_pred or args.save_vis:
                save_mask(os.path.join(pred_dir, img_id + "_Pred.png"), pred_np)
                save_mask(os.path.join(gt_dir, img_id + "_GT.png"), gt_np)
            if args.save_vis:
                raw_rgb = image_to_uint8(images[bi])
                panel = make_panel(raw_rgb, gt_np, pred_np, prob_np, img_id, iou_i)
                panel.save(os.path.join(vis_dir, img_id + "_fuse.png"))
        global_idx += b

    _, iou = iou_metric.get()
    _, niou = niou_metric.get()
    tpr, fpr, recall, precision = roc.get()
    fa, pd = pdfa.get()
    idx05 = int(round(args.threshold * 10))
    metrics = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "IoU": float(iou),
        "nIoU": float(niou),
        "PD@threshold": float(pd[idx05]),
        "FA@threshold": float(fa[idx05]),
        "TPR": tpr.tolist(),
        "FPR": fpr.tolist(),
        "Recall": recall.tolist(),
        "Precision": precision.tolist(),
        "PD_curve": pd.tolist(),
        "FA_curve": fa.tolist(),
    }
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.save_dir, f"{args.dataset}_test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"IoU:\n{iou}\n")
        f.write(f"nIoU:\n{niou}\n")
        f.write(f"PD@{args.threshold}:\n{pd[idx05]}\n")
        f.write(f"FA@{args.threshold}:\n{fa[idx05]}\n")
        f.write(f"TPR:\n{tpr}\n")
        f.write(f"FPR:\n{fpr}\n")
        f.write(f"Recall:\n{recall}\n")
        f.write(f"Precision:\n{precision}\n")

    print("========== Test Results ==========")
    print(f"IoU  : {iou:.6f}")
    print(f"nIoU : {niou:.6f}")
    print(f"PD@{args.threshold}: {pd[idx05]:.6f}")
    print(f"FA@{args.threshold}: {fa[idx05]:.8f}")
    print(f"Saved to: {args.save_dir}")
    return metrics


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_dir, val_meta, img_size, split_method = resolve_dataset_paths(args)
    ids = read_ids(val_meta)
    print(f"dataset_dir: {dataset_dir}")
    print(f"val_meta   : {val_meta} ({len(ids)} samples)")
    print(f"img_size   : {img_size}")

    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    valset = TestSetLoader(
        dataset_dir,
        img_id=ids,
        base_size=img_size,
        crop_size=img_size,
        transform=input_transform,
        form=args.form,
        return_id=True,
    )
    loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

    model = build_model(args, img_size, device)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"# model parameters: {params_m:.2f} M")
    load_checkpoint(model, args.checkpoint, device)
    test(model, loader, device, args, ids)


if __name__ == "__main__":
    main(parse_args())
