"""Fix top-right-corner shortcut learning, then rerun Grad-CAM error analysis.

This script fine-tunes the V3 EfficientNet-B4 classifier with two targeted fixes:

1. CropNonWhite: crop around the minifigure/object and remove large white margins.
2. RandomTopRightErase: randomly erase the top-right corner during training.

Then it evaluates the fixed model and writes a wrong-prediction Grad-CAM report.

Recommended for V3 focal checkpoint:

python3 standalone_fix_corner_shortcut_and_gradcam.py \
  --data-dir "/home/yourname/lego_project/Assignment 2_final" \
  --checkpoint "/home/yourname/lego_project/Assignment 2_final/optionB_v3_results/best_model_focal.pth" \
  --variant focal \
  --epochs 2 \
  --batch-size 8 \
  --out-dir results_v3_corner_fix

Outputs:
  results_v3_corner_fix/
    fixed_corner_model.pth
    training_history.csv
    fixed_wrong_predictions_gradcam.csv
    fixed_wrong_predictions_gradcam_report.html
    fixed_wrong_gradcam_examples/
    SUMMARY.txt

Note: this script supports fine-tuning normal classifier variants (`focal` and
`supcon`). ArcFace requires a separate margin-head training path, so this script
will stop if `--variant arcface` is used.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMG_SIZE = 380
SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


MERGE_MAP = {
    "The LEGO NINJAGO Movie": "NINJAGO",
    "The LEGO Movie": "LEGO Movies", "The LEGO Movie 2": "LEGO Movies",
    "DC Super Hero Girls": "Super Heroes", "Spider-Man": "Super Heroes", "Batman I": "Super Heroes",
    "Elves": "Friends & Fantasy", "Friends": "Friends & Fantasy",
    "DUPLO": "Preschool", "Primo": "Preschool", "Belville": "Preschool", "Scala": "Preschool",
    "For Juniors": "Preschool", "Homemaker": "Preschool", "Fabuland": "Preschool",
    "Pirates of the Caribbean": "Pirates",
    "Castle": "Castle & Medieval", "Vikings": "Castle & Medieval", "Ninja": "Castle & Medieval",
    "NEXO KNIGHTS": "Castle & Medieval",
    "Space": "Space & Sci-Fi", "Aquazone": "Space & Sci-Fi", "Alpha Team": "Space & Sci-Fi",
    "Exo-Force": "Space & Sci-Fi", "Power Miners": "Space & Sci-Fi", "Rock Raiders": "Space & Sci-Fi",
    "Agents": "Space & Sci-Fi", "Ultra Agents": "Space & Sci-Fi", "Atlantis": "Space & Sci-Fi",
    "Adventurers": "Adventure", "Indiana Jones": "Adventure", "Pharaoh's Quest": "Adventure",
    "Dino Attack": "Adventure", "Dino": "Adventure", "Western": "Adventure",
    "SPEED CHAMPIONS": "Racing & Vehicles", "Racers": "Racing & Vehicles",
    "World Racers": "Racing & Vehicles", "Speed Racer": "Racing & Vehicles", "Cars": "Racing & Vehicles",
    "LEGENDS OF CHIMA": "Fantasy", "DREAMZzz": "Fantasy", "Hidden Side": "Fantasy",
    "Monster Fighters": "Fantasy",
    "The Hobbit and The Lord of the Rings": "Middle-Earth",
    "Teenage Mutant Ninja Turtles": "Licensed Entertainment",
    "SpongeBob SquarePants": "Licensed Entertainment", "Scooby-Doo": "Licensed Entertainment",
    "The Simpsons": "Licensed Entertainment", "Ghostbusters": "Licensed Entertainment",
    "Stranger Things": "Licensed Entertainment", "The Angry Birds Movie": "Licensed Entertainment",
    "Trolls World Tour": "Licensed Entertainment", "The Incredibles": "Licensed Entertainment",
    "Toy Story": "Licensed Entertainment", "Wednesday": "Licensed Entertainment",
    "Wicked": "Licensed Entertainment", "Despicable Me and Minions": "Licensed Entertainment",
    "The Lone Ranger": "Licensed Entertainment", "Prince of Persia": "Licensed Entertainment",
    "Island Xtreme Stunts": "Licensed Entertainment", "The Powerpuff Girls": "Licensed Entertainment",
    "Back to the Future": "Licensed Entertainment", "Friends TV Series": "Licensed Entertainment",
    "Queer Eye": "Licensed Entertainment", "Lightyear": "Licensed Entertainment",
    "Bluey": "Licensed Entertainment", "Gabby's Dollhouse": "Licensed Entertainment",
    "Dune": "Licensed Entertainment", "Star Trek": "Licensed Entertainment",
    "Overwatch": "Video Games", "Fortnite": "Video Games", "Sonic the Hedgehog": "Video Games",
    "Horizon": "Video Games", "The Legend of Zelda": "Video Games", "Animal Crossing": "Video Games",
    "Avatar The Last Airbender": "Video Games", "Avatar": "Video Games", "One Piece": "Video Games",
    "Collectible Minifigures": "Collectible & Special", "Holiday & Event": "Collectible & Special",
    "Dimensions": "Collectible & Special", "Vidiyo": "Collectible & Special", "Games": "Collectible & Special",
    "LEGO Ideas": "LEGO Promotional", "BrickLink Designer Program": "LEGO Promotional",
    "LEGO Brand": "LEGO Promotional", "LEGOLAND": "LEGO Promotional", "LEGOLAND Parks": "LEGO Promotional",
    "Promotional": "LEGO Promotional", "Studios": "LEGO Promotional", "FIRST LEGO League": "LEGO Promotional",
    "Educational & Dacta": "Education & Technical", "BIONICLE": "Education & Technical",
    "Hero Factory": "Education & Technical", "Technic": "Education & Technical",
    "Basic": "Education & Technical", "FreeStyle": "Education & Technical",
    "Master Builder Academy": "Education & Technical", "Building Bigger Thinking": "Education & Technical",
    "Discovery": "Education & Technical", "Quatro": "Education & Technical",
    "Universe": "Education & Technical", "Fusion": "Education & Technical",
    "Nike": "Education & Technical", "Architecture": "Education & Technical",
    "Clikits": "Education & Technical", "Unikitty!": "Education & Technical",
}


def split_town(subcategory: str) -> str:
    sub = str(subcategory).lower()
    if "police" in sub: return "Town - Police"
    if "fire" in sub: return "Town - Fire"
    if "airport" in sub: return "Town - Airport"
    if "hospital" in sub or "rescue" in sub: return "Town - Rescue"
    if "space" in sub: return "Town - Space"
    if "race" in sub or "stuntz" in sub: return "Town - Racing"
    if "coast guard" in sub: return "Town - Coast Guard"
    if "construction" in sub: return "Town - Construction"
    if any(x in sub for x in ["arctic", "jungle", "volcano", "ocean", "deep sea", "exploration"]):
        return "Town - Exploration"
    return "Town - General"


def load_records(data_dir: Path):
    with open(data_dir / "minifigs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for d in data:
        rel = d.get("img_local_path")
        if rel and (data_dir / rel).exists():
            d = dict(d)
            d["target_category"] = split_town(d.get("subcategory", "")) if d.get("category") == "Town" else MERGE_MAP.get(d.get("category"), d.get("category"))
            records.append(d)
    cat_names = sorted({r["target_category"] for r in records})
    cat2idx = {c: i for i, c in enumerate(cat_names)}
    idx2cat = {i: c for c, i in cat2idx.items()}
    labels = [cat2idx[r["target_category"]] for r in records]
    return records, labels, cat2idx, idx2cat


def make_splits(records, labels):
    counts = Counter(labels)
    small_idx = [i for i, y in enumerate(labels) if counts[y] < 7]
    big_idx = [i for i, y in enumerate(labels) if counts[y] >= 7]
    big_records = [records[i] for i in big_idx]
    big_labels = [labels[i] for i in big_idx]
    train_big, temp_records, train_y_big, temp_y = train_test_split(
        big_records, big_labels, test_size=0.3, random_state=SEED, stratify=big_labels
    )
    val_records, test_records, val_y, test_y = train_test_split(
        temp_records, temp_y, test_size=0.5, random_state=SEED, stratify=temp_y
    )
    train_records = train_big + [records[i] for i in small_idx]
    train_y = train_y_big + [labels[i] for i in small_idx]
    return train_records, train_y, val_records, val_y, test_records, test_y


class CropNonWhite:
    def __init__(self, threshold=245, padding=12, min_area_frac=0.01):
        self.threshold = threshold
        self.padding = padding
        self.min_area_frac = min_area_frac

    def __call__(self, img):
        arr = np.asarray(img.convert("RGB"))
        mask = np.any(arr < self.threshold, axis=2)
        if mask.mean() < self.min_area_frac:
            return img
        ys, xs = np.where(mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(img.width, x2 + self.padding)
        y2 = min(img.height, y2 + self.padding)
        return img.crop((x1, y1, x2, y2))


class RandomTopRightErase:
    def __init__(self, p=0.6, min_frac=0.08, max_frac=0.24, fill=(255, 255, 255)):
        self.p = p
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.fill = fill

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = img.copy()
        w, h = img.size
        fw = random.uniform(self.min_frac, self.max_frac)
        fh = random.uniform(self.min_frac, self.max_frac)
        box_w = int(w * fw)
        box_h = int(h * fh)
        draw = ImageDraw.Draw(img)
        draw.rectangle([w - box_w, 0, w, box_h], fill=self.fill)
        return img


class PadToSquare:
    def __call__(self, img):
        side = max(img.size)
        return ImageOps.pad(img, (side, side), color=(255, 255, 255))


train_transform = transforms.Compose([
    CropNonWhite(),
    RandomTopRightErase(p=0.6),
    PadToSquare(),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=255),
    transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

fixed_eval_transform = transforms.Compose([
    CropNonWhite(),
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class MinifigDataset(Dataset):
    def __init__(self, data_dir, records, labels, transform):
        self.data_dir = data_dir
        self.records = records
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img = Image.open(self.data_dir / self.records[idx]["img_local_path"]).convert("RGB")
        return self.transform(img), int(self.labels[idx])


def build_v3_model(num_classes: int):
    backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
    in_features = backbone.num_features
    classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return nn.Sequential(backbone, classifier)


def load_classifier_checkpoint(checkpoint_path, num_classes, device):
    model = build_v3_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    return model.to(device)


def freeze_for_shortcut_fix(model, unfreeze_last_block=False):
    for p in model.parameters():
        p.requires_grad = False
    for p in model[1].parameters():
        p.requires_grad = True
    if unfreeze_last_block:
        for p in model[0].blocks[-1].parameters():
            p.requires_grad = True
        for p in model[0].conv_head.parameters():
            p.requires_grad = True
        for p in model[0].bn2.parameters():
            p.requires_grad = True


def train_fix(model, data_dir, train_records, train_labels, val_records, val_labels, device, epochs, batch_size, lr):
    freeze_for_shortcut_fix(model, unfreeze_last_block=False)
    counts = Counter(train_labels)
    weights = [1.0 / math.sqrt(counts[y]) for y in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_ds = MinifigDataset(data_dir, train_records, train_labels, train_transform)
    val_ds = MinifigDataset(data_dir, val_records, val_labels, fixed_eval_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = torch.zeros(max(train_labels) + 1)
    for i in range(len(class_weights)):
        class_weights[i] = 1.0 / math.sqrt(counts.get(i, 1))
    class_weights = (class_weights / class_weights.sum() * len(class_weights)).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
            total_correct += int((logits.argmax(1) == y).sum().item())
            total += x.size(0)

        val_preds, val_true = predict_dataset(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": total_correct / max(total, 1),
            "val_accuracy": accuracy_score(val_true, val_preds),
            "val_macro_f1": f1_score(val_true, val_preds, average="macro", zero_division=0),
        }
        history.append(row)
        print(
            f"epoch {row['epoch']}/{epochs}: loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f}",
            flush=True,
        )
    return pd.DataFrame(history)


def predict_dataset(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            true.extend(y.numpy().tolist())
    return np.array(preds), np.array(true)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def remove(self):
        self.fh.remove()

    def __call__(self, x):
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        # The backbone is frozen during shortcut-fix fine-tuning. Grad-CAM still
        # needs gradients through the feature maps, so make the input require
        # grad during explanation.
        x = x.detach().clone().requires_grad_(True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(1).item())
        logits[:, pred].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM failed to capture activations/gradients. "
                "Check that the target layer is used in the model forward pass."
            )
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam, pred, float(probs.max(1).values.item())


def unnormalize(x):
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(STD) + np.array(MEAN)
    return np.clip(arr, 0, 1)


def overlay_cam(rgb, cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.55 * rgb + 0.45 * heat, 0, 1)


def attention_region_stats(rgb, cam):
    h, w = cam.shape
    border = np.zeros_like(cam, dtype=bool)
    my, mx = max(1, int(0.08 * h)), max(1, int(0.08 * w))
    border[:my, :] = True
    border[-my:, :] = True
    border[:, :mx] = True
    border[:, -mx:] = True
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    background = (hsv[..., 2] / 255.0 > 0.82) & (hsv[..., 1] / 255.0 < 0.18)
    foreground = ~background
    total = float(cam.sum() + 1e-9)
    border_attention = float(cam[border].sum() / total)
    background_attention = float(cam[background].sum() / total)
    foreground_attention = float(cam[foreground].sum() / total)
    peak = cam >= np.quantile(cam, 0.90)
    peak_border = float((peak & border).sum() / max(peak.sum(), 1))
    peak_background = float((peak & background).sum() / max(peak.sum(), 1))
    if background_attention > 0.45 or peak_background > 0.45:
        diagnosis = "likely_background_shortcut"
    elif border_attention > 0.25 or peak_border > 0.30:
        diagnosis = "likely_border_shortcut"
    elif foreground_attention > 0.60:
        diagnosis = "focuses_on_figure_but_wrong"
    else:
        diagnosis = "mixed_attention"
    return {
        "border_attention": border_attention,
        "background_attention": background_attention,
        "foreground_attention": foreground_attention,
        "peak_border_fraction": peak_border,
        "peak_background_fraction": peak_background,
        "attention_diagnosis": diagnosis,
    }


def save_png(path, arr):
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def image_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def write_html(out_dir, rows, max_items):
    cards = []
    for row in rows[:max_items]:
        cards.append(f"""
        <article class="card">
          <div class="imgs">
            <figure><img src="data:image/png;base64,{image_to_b64(Path(row['original_path']))}"><figcaption>Fixed preprocessing input</figcaption></figure>
            <figure><img src="data:image/png;base64,{image_to_b64(Path(row['gradcam_path']))}"><figcaption>Grad-CAM after shortcut fix</figcaption></figure>
          </div>
          <div class="meta">
            <h2>{html.escape(row['image_id'])}</h2>
            <p><b>True:</b> {html.escape(row['true_category'])}</p>
            <p><b>Predicted:</b> {html.escape(row['pred_category'])} ({row['confidence']:.2%})</p>
            <p><b>Attention diagnosis:</b> <code>{html.escape(row['attention_diagnosis'])}</code></p>
            <p><b>Background attention:</b> {row['background_attention']:.2%}</p>
            <p><b>Border attention:</b> {row['border_attention']:.2%}</p>
            <p><b>Foreground attention:</b> {row['foreground_attention']:.2%}</p>
          </div>
        </article>
        """)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fixed Model Wrong Prediction Grad-CAM</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17202a}}
header{{padding:28px 36px;border-bottom:1px solid #d7dbdd}}h1{{margin:0 0 6px;font-size:24px}}.sub{{margin:0;color:#5d6d7e}}
main{{padding:24px 36px 40px;display:grid;gap:18px}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}
.metric{{background:#f4f6f7;border-radius:6px;padding:12px}}.metric b{{display:block;font-size:12px;color:#5d6d7e;margin-bottom:4px}}
.card{{border-top:1px solid #d7dbdd;padding-top:18px;display:grid;grid-template-columns:minmax(300px,2fr) minmax(240px,1fr);gap:18px}}
.imgs{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0}}img{{width:100%;border:1px solid #d7dbdd;border-radius:6px;object-fit:contain;background:white}}
figcaption{{margin-top:6px;color:#5d6d7e;font-size:13px}}h2{{margin:0 0 10px;font-size:17px}}p{{margin:7px 0}}code{{background:#eef2f3;padding:2px 5px;border-radius:4px}}
@media(max-width:820px){{.card{{grid-template-columns:1fr}}.imgs{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Fixed Model Wrong Prediction Grad-CAM</h1><p class="sub">After CropNonWhite + RandomTopRightErase fine-tuning.</p></header>
<main><section class="summary"><div class="metric"><b>Wrong predictions after fix</b>{len(rows)}</div><div class="metric"><b>Showing</b>{min(len(rows), max_items)}</div><div class="metric"><b>CSV</b>fixed_wrong_predictions_gradcam.csv</div></section>
{''.join(cards)}</main></body></html>"""
    out_path = out_dir / "fixed_wrong_predictions_gradcam_report.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


def run_fixed_wrong_gradcam(model, data_dir, test_records, test_labels, idx2cat, device, out_dir, max_wrong, html_items):
    examples_dir = out_dir / "fixed_wrong_gradcam_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    cammer = GradCAM(model, model[0].conv_head)
    rows = []
    for rec, label in zip(test_records, test_labels):
        image_path = data_dir / rec["img_local_path"]
        image = Image.open(image_path).convert("RGB")
        x = fixed_eval_transform(image).unsqueeze(0).to(device)
        cam, pred, conf = cammer(x)
        if pred == label:
            continue
        rgb = unnormalize(x[0])
        overlay = overlay_cam(rgb, cam)
        stats = attention_region_stats(rgb, cam)
        stem = f"{len(rows):04d}_{Path(rec['img_local_path']).stem}"
        original_path = examples_dir / f"{stem}_fixed_input.png"
        gradcam_path = examples_dir / f"{stem}_fixed_gradcam.png"
        save_png(original_path, rgb)
        save_png(gradcam_path, overlay)
        rows.append({
            "image_id": Path(rec["img_local_path"]).name,
            "true_label": int(label),
            "true_category": idx2cat[int(label)],
            "pred_label": int(pred),
            "pred_category": idx2cat[int(pred)],
            "confidence": float(conf),
            "image_path": str(image_path),
            "original_path": str(original_path),
            "gradcam_path": str(gradcam_path),
            **stats,
        })
        if len(rows) >= max_wrong:
            break
    cammer.remove()
    df = pd.DataFrame(rows)
    csv_path = out_dir / "fixed_wrong_predictions_gradcam.csv"
    df.to_csv(csv_path, index=False)
    html_path = write_html(out_dir, rows, html_items)
    return df, csv_path, html_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=["focal", "supcon", "arcface"], default="focal")
    parser.add_argument("--out-dir", type=Path, default=Path("results_v3_corner_fix"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-wrong", type=int, default=200)
    parser.add_argument("--html-items", type=int, default=80)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    if args.variant == "arcface":
        raise SystemExit("This shortcut-fix trainer supports focal/supcon classifier checkpoints, not ArcFace.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    records, labels, cat2idx, idx2cat = load_records(args.data_dir)
    train_records, train_labels, val_records, val_labels, test_records, test_labels = make_splits(records, labels)
    model = load_classifier_checkpoint(args.checkpoint, len(cat2idx), device)

    print(f"Device: {device}")
    print(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
    history = train_fix(model, args.data_dir, train_records, train_labels, val_records, val_labels, device, args.epochs, args.batch_size, args.lr)
    history.to_csv(args.out_dir / "training_history.csv", index=False)

    torch.save({"model": model.state_dict(), "preprocessing": "CropNonWhite+PadToSquare+Resize380"}, args.out_dir / "fixed_corner_model.pth")

    test_ds = MinifigDataset(args.data_dir, test_records, test_labels, fixed_eval_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_preds, test_true = predict_dataset(model, test_loader, device)
    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)

    wrong_df, csv_path, html_path = run_fixed_wrong_gradcam(
        model, args.data_dir, test_records, test_labels, idx2cat, device, args.out_dir, args.max_wrong, args.html_items
    )

    diagnosis = wrong_df["attention_diagnosis"].value_counts().to_string() if not wrong_df.empty else "No wrong predictions found."
    summary = (
        f"Device: {device}\n"
        f"Fixed model test accuracy: {test_acc:.4f}\n"
        f"Fixed model test macro F1: {test_f1:.4f}\n"
        f"Wrong predictions analyzed after fix: {len(wrong_df)}\n\n"
        f"Attention diagnosis counts:\n{diagnosis}\n"
    )
    (args.out_dir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote fixed checkpoint: {args.out_dir / 'fixed_corner_model.pth'}")
    print(f"Wrote Grad-CAM CSV: {csv_path}")
    print(f"Wrote Grad-CAM HTML: {html_path}")


if __name__ == "__main__":
    main()
