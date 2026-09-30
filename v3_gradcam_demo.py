"""Grad-CAM XAI demo for the final V3 EfficientNet-B4 model.

This runs on clean images before IQA/degradation analysis. It answers:
    Which parts of the image does the final classifier focus on?

Example:
    python3 v3_gradcam_demo.py \
      --data-dir "/path/to/Assignment 2_final" \
      --checkpoint "/path/to/optionB_v3_results/best_model_focal.pth" \
      --variant focal \
      --num-images 24
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
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


def make_test_split(records, labels):
    counts = Counter(labels)
    big_idx = [i for i, y in enumerate(labels) if counts[y] >= 7]
    big_records = [records[i] for i in big_idx]
    big_labels = [labels[i] for i in big_idx]
    _, temp_records, _, temp_y = train_test_split(
        big_records, big_labels, test_size=0.3, random_state=SEED, stratify=big_labels
    )
    _, test_records, _, test_y = train_test_split(
        temp_records, temp_y, test_size=0.5, random_state=SEED, stratify=temp_y
    )
    return test_records, test_y


class PadToSquare:
    def __call__(self, img):
        side = max(img.size)
        return ImageOps.pad(img, (side, side), color=(255, 255, 255))


eval_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale


class ArcFaceInference(nn.Module):
    def __init__(self, embedding_model, arcface_head):
        super().__init__()
        self.embedding_model = embedding_model
        self.arcface_head = arcface_head

    def forward(self, x):
        return self.arcface_head(self.embedding_model(x))


def build_v3_model(variant: str, num_classes: int):
    backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
    in_features = backbone.num_features

    if variant == "arcface":
        classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        model = nn.Sequential(backbone, classifier)
        arcface = ArcFaceHead(512, num_classes, scale=30.0, margin=0.5)
        return model, arcface

    classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    model = nn.Sequential(backbone, classifier)
    return model, None


def load_v3(checkpoint_path: Path, variant: str, num_classes: int, device):
    model, arcface = build_v3_model(variant, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        if variant == "arcface":
            if "arcface_head" not in checkpoint:
                raise ValueError("ArcFace checkpoint is missing `arcface_head`.")
            arcface.load_state_dict(checkpoint["arcface_head"])
    else:
        model.load_state_dict(checkpoint)

    if variant == "arcface":
        wrapped = ArcFaceInference(model, arcface)
        wrapped.to(device).eval()
        return wrapped, model[0]

    model.to(device).eval()
    return model, model[0]


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self._save_activation)
        self.bh = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fh.remove()
        self.bh.remove()

    def __call__(self, x, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        if target_class is None:
            target_class = int(probs.argmax(1).item())
        score = logits[:, target_class].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam, int(probs.argmax(1).item()), float(probs.max(1).values.item())


def unnormalize(x):
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(STD) + np.array(MEAN)
    return np.clip(arr, 0, 1)


def overlay_cam(rgb, cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.55 * rgb + 0.45 * heat, 0, 1)


def save_pdf_report(out_dir: Path, df: pd.DataFrame, grid_path: Path, variant: str):
    pdf_path = out_dir / "v3_gradcam_report.pdf"
    accuracy = float(df["correct"].mean()) if "correct" in df else float("nan")

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "V3 Grad-CAM XAI Demo", fontsize=18, fontweight="bold")
        paragraphs = [
            f"This report applies Grad-CAM to the final V3 {variant} classifier on clean LEGO minifigure images before the IQA/degradation pipeline.",
            "Grad-CAM highlights the spatial regions that contributed most to the predicted class. In a reliable visual classifier, the highlighted areas should usually align with meaningful parts of the minifigure, such as torso print, head, clothing, helmet, or accessories.",
            "If heatmaps focus on blank background, image borders, catalogue artifacts, or unrelated regions, this suggests shortcut learning. That weakness should be addressed before robustness testing and medical-style image quality assessment.",
            f"Demo-set accuracy: {accuracy:.3f}. This is only for the sampled images in this XAI demo, not the full test-set result.",
        ]
        y = 0.86
        for para in paragraphs:
            for line in textwrap.wrap(para, 88):
                fig.text(0.08, y, line, fontsize=11)
                y -= 0.028
            y -= 0.022
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        img = Image.open(grid_path).convert("RGB")
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Grad-CAM Demo Grid", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing minifigs.json and images/.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="V3 checkpoint path.")
    parser.add_argument("--variant", choices=["focal", "arcface", "supcon"], default="focal")
    parser.add_argument("--out-dir", type=Path, default=Path("results_v3_xai"))
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.out_dir / "gradcam_examples"
    examples_dir.mkdir(exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    records, labels, cat2idx, idx2cat = load_records(args.data_dir)
    test_records, test_labels = make_test_split(records, labels)
    sample_records = test_records[:args.num_images]
    sample_labels = test_labels[:args.num_images]

    model, backbone = load_v3(args.checkpoint, args.variant, len(cat2idx), device)
    target_layer = backbone.conv_head
    cammer = GradCAM(model, target_layer)

    rows = []
    panels = []
    for i, (rec, label) in enumerate(zip(sample_records, sample_labels)):
        path = args.data_dir / rec["img_local_path"]
        img = Image.open(path).convert("RGB")
        x = eval_transform(img).unsqueeze(0).to(device)
        cam, pred, conf = cammer(x)
        rgb = unnormalize(x[0])
        overlay = overlay_cam(rgb, cam)
        correct = int(pred == label)

        stem = f"{i:03d}_{Path(rec['img_local_path']).stem}"
        overlay_path = examples_dir / f"{stem}_gradcam.png"
        original_path = examples_dir / f"{stem}_original.png"
        plt.imsave(overlay_path, overlay)
        plt.imsave(original_path, rgb)

        rows.append({
            "image_id": Path(rec["img_local_path"]).name,
            "true_label": int(label),
            "true_category": idx2cat[label],
            "pred_label": int(pred),
            "pred_category": idx2cat[pred],
            "confidence": conf,
            "correct": correct,
            "image_path": str(path),
            "gradcam_path": str(overlay_path),
        })
        panels.append((rgb, overlay, idx2cat[label], idx2cat[pred], conf, correct))

    cammer.remove()
    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "v3_xai_predictions.csv"
    df.to_csv(csv_path, index=False)

    n = min(len(panels), 12)
    fig, axes = plt.subplots(n, 2, figsize=(8, max(3 * n, 6)))
    if n == 1:
        axes = np.array([axes])
    for row in range(n):
        rgb, overlay, true_cat, pred_cat, conf, correct = panels[row]
        axes[row, 0].imshow(rgb)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"Original\nTrue: {true_cat}", fontsize=9)
        axes[row, 1].imshow(overlay)
        axes[row, 1].axis("off")
        mark = "correct" if correct else "wrong"
        axes[row, 1].set_title(f"Grad-CAM\nPred: {pred_cat} ({conf:.2f}) - {mark}", fontsize=9)
    fig.tight_layout()
    grid_path = args.out_dir / "v3_gradcam_grid.png"
    fig.savefig(grid_path, dpi=180)
    plt.close(fig)

    pdf_path = save_pdf_report(args.out_dir, df, grid_path, args.variant)
    print(f"Device: {device}")
    print(f"Wrote predictions: {csv_path}")
    print(f"Wrote Grad-CAM grid: {grid_path}")
    print(f"Wrote PDF report: {pdf_path}")


if __name__ == "__main__":
    main()
