"""Grad-CAM error analysis for all wrongly predicted V3 test images.

This script runs the trained V3 model over the recreated test split, keeps the
wrong predictions, generates Grad-CAM overlays, and estimates whether attention
falls on likely unimportant regions such as image borders or white background.

Example:
    python3 v3_wrong_predictions_gradcam.py \
      --data-dir "/home/yourname/lego_project/Assignment 2_final" \
      --checkpoint "/home/yourname/lego_project/Assignment 2_final/optionB_v3_results/best_model_focal.pth" \
      --variant focal \
      --out-dir results_v3_wrong_gradcam
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def image_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def save_png(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def attention_region_stats(rgb: np.ndarray, cam: np.ndarray) -> dict[str, float | str]:
    """Estimate whether Grad-CAM attention is on figure, background, or border.

    This is a heuristic, not a ground-truth segmentation. BrickLink images often
    have white backgrounds, so saturation/brightness thresholds work reasonably
    as a first-pass foreground/background proxy.
    """
    h, w = cam.shape
    border = np.zeros_like(cam, dtype=bool)
    margin_y = max(1, int(0.08 * h))
    margin_x = max(1, int(0.08 * w))
    border[:margin_y, :] = True
    border[-margin_y:, :] = True
    border[:, :margin_x] = True
    border[:, -margin_x:] = True

    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0

    # White/near-white background proxy.
    background = (value > 0.82) & (saturation < 0.18)
    foreground = ~background

    total_attention = float(cam.sum() + 1e-9)
    border_attention = float(cam[border].sum() / total_attention)
    background_attention = float(cam[background].sum() / total_attention)
    foreground_attention = float(cam[foreground].sum() / total_attention)

    peak_mask = cam >= np.quantile(cam, 0.90)
    peak_border = float((peak_mask & border).sum() / max(peak_mask.sum(), 1))
    peak_background = float((peak_mask & background).sum() / max(peak_mask.sum(), 1))

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


def write_html_report(out_dir: Path, rows: list[dict], max_items: int) -> Path:
    cards = []
    for row in rows[:max_items]:
        cards.append(f"""
        <article class="card">
          <div class="imgs">
            <figure>
              <img src="data:image/png;base64,{image_to_b64(Path(row['original_path']))}" alt="original">
              <figcaption>Original</figcaption>
            </figure>
            <figure>
              <img src="data:image/png;base64,{image_to_b64(Path(row['gradcam_path']))}" alt="gradcam">
              <figcaption>Grad-CAM overlay</figcaption>
            </figure>
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
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V3 Wrong Prediction Grad-CAM Analysis</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; }}
    header {{ padding: 28px 36px; border-bottom: 1px solid #d7dbdd; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .sub {{ margin: 0; color: #5d6d7e; }}
    main {{ padding: 24px 36px 40px; display: grid; gap: 18px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .metric {{ background: #f4f6f7; border-radius: 6px; padding: 12px; }}
    .metric b {{ display: block; font-size: 12px; color: #5d6d7e; margin-bottom: 4px; }}
    .card {{ border-top: 1px solid #d7dbdd; padding-top: 18px; display: grid; grid-template-columns: minmax(300px, 2fr) minmax(240px, 1fr); gap: 18px; }}
    .imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border: 1px solid #d7dbdd; border-radius: 6px; object-fit: contain; background: white; }}
    figcaption {{ margin-top: 6px; color: #5d6d7e; font-size: 13px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; }}
    p {{ margin: 7px 0; }}
    code {{ background: #eef2f3; padding: 2px 5px; border-radius: 4px; }}
    @media (max-width: 820px) {{
      .card {{ grid-template-columns: 1fr; }}
      .imgs {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>V3 Wrong Prediction Grad-CAM Analysis</h1>
    <p class="sub">Wrongly predicted test images with heuristic attention analysis for background and border focus.</p>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><b>Total wrong predictions</b>{len(rows)}</div>
      <div class="metric"><b>Showing</b>{min(len(rows), max_items)}</div>
      <div class="metric"><b>CSV</b>wrong_predictions_gradcam.csv</div>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    out_path = out_dir / "wrong_predictions_gradcam_report.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=["focal", "arcface", "supcon"], default="focal")
    parser.add_argument("--out-dir", type=Path, default=Path("results_v3_wrong_gradcam"))
    parser.add_argument("--max-wrong", type=int, default=200, help="Maximum wrong predictions to analyze.")
    parser.add_argument("--html-items", type=int, default=80, help="Maximum examples embedded in HTML.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.out_dir / "wrong_gradcam_examples"
    examples_dir.mkdir(exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    records, labels, _, idx2cat = load_records(args.data_dir)
    test_records, test_labels = make_test_split(records, labels)
    model, backbone = load_v3(args.checkpoint, args.variant, len(idx2cat), device)
    cammer = GradCAM(model, backbone.conv_head)

    rows = []
    seen_wrong = 0
    for i, (rec, label) in enumerate(zip(test_records, test_labels)):
        image_path = args.data_dir / rec["img_local_path"]
        image = Image.open(image_path).convert("RGB")
        x = eval_transform(image).unsqueeze(0).to(device)
        cam, pred, conf = cammer(x)

        if pred == label:
            continue

        rgb = unnormalize(x[0])
        overlay = overlay_cam(rgb, cam)
        stats = attention_region_stats(rgb, cam)

        stem = f"{seen_wrong:04d}_{Path(rec['img_local_path']).stem}"
        original_path = examples_dir / f"{stem}_original.png"
        gradcam_path = examples_dir / f"{stem}_gradcam.png"
        save_png(original_path, rgb)
        save_png(gradcam_path, overlay)

        row = {
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
        }
        rows.append(row)
        seen_wrong += 1

        if seen_wrong >= args.max_wrong:
            break

        if seen_wrong % 25 == 0:
            print(f"analyzed {seen_wrong} wrong predictions", flush=True)

    cammer.remove()
    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "wrong_predictions_gradcam.csv"
    df.to_csv(csv_path, index=False)
    html_path = write_html_report(args.out_dir, rows, args.html_items)

    if not df.empty:
        summary = df["attention_diagnosis"].value_counts().to_string()
    else:
        summary = "No wrong predictions found."

    (args.out_dir / "SUMMARY.txt").write_text(
        f"Device: {device}\nWrong predictions analyzed: {len(df)}\n\nAttention diagnosis counts:\n{summary}\n",
        encoding="utf-8",
    )
    print(f"Device: {device}")
    print(f"Wrong predictions analyzed: {len(df)}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote HTML report: {html_path}")
    print(summary)


if __name__ == "__main__":
    main()
