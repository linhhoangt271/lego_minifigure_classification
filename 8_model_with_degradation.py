"""Degradation/IQA benchmark for the fixed V3 model.

Use this after `standalone_fix_corner_shortcut_and_gradcam.py` has produced:
    results_v3_corner_fix_v2/fixed_corner_model.pth

The script evaluates how controlled image-quality degradation affects the fixed
classifier. It writes:
    - degradation_results.csv
    - degradation_summary.html
    - accuracy_by_degradation.png
    - confidence_by_degradation.png
    - quality_vs_correct.png
    - SUMMARY.txt

Example:
python3 standalone_fixed_model_degradation_benchmark.py \
  --data-dir "/home/test/barco" \
  --checkpoint "/home/test/barco/results_v3_corner_fix_v2/fixed_corner_model.pth" \
  --out-dir results_fixed_model_degradation \
  --batch-size 16
"""
from __future__ import annotations

import argparse
import html
import json
import math
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
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
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
    classes = sorted({r["target_category"] for r in records})
    cat2idx = {c: i for i, c in enumerate(classes)}
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


def build_model(num_classes):
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


def load_model(checkpoint_path, num_classes, device):
    model = build_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model.to(device).eval()


def disk_kernel(radius: float):
    r = int(math.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (xx ** 2 + yy ** 2 <= radius ** 2).astype(np.float32)
    if k.sum() == 0:
        k[r, r] = 1
    return k / k.sum()


def motion_kernel(length: int, angle: float):
    length = int(length) | 1
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, m, (length, length))
    return k / max(k.sum(), 1e-9)


PARAMS = {
    "defocus_blur": [0, 1.5, 3.0, 5.0],
    "motion_blur": [0, 5, 11, 21],
    "low_brightness": [1.0, 0.75, 0.50, 0.30],
    "low_contrast": [1.0, 0.70, 0.45, 0.25],
    "jpeg_compression": [95, 55, 30, 12],
    "noise": [0, 8, 18, 35],
    "occlusion": [0.0, 0.08, 0.18, 0.35],
    "color_cast": [0.0, 0.06, 0.13, 0.25],
}


def degrade(img: Image.Image, kind: str, severity: int, rng):
    if kind == "clean" or severity == 0:
        return img.copy()
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    p = PARAMS[kind][severity]
    if kind == "defocus_blur":
        out = cv2.filter2D(bgr, -1, disk_kernel(float(p)), borderType=cv2.BORDER_REPLICATE)
    elif kind == "motion_blur":
        out = cv2.filter2D(bgr, -1, motion_kernel(int(p), rng.uniform(0, 180)), borderType=cv2.BORDER_REPLICATE)
    elif kind == "low_brightness":
        out = np.clip(bgr.astype(np.float32) * float(p), 0, 255).astype(np.uint8)
    elif kind == "low_contrast":
        out = np.clip((bgr.astype(np.float32) - 127.5) * float(p) + 127.5, 0, 255).astype(np.uint8)
    elif kind == "jpeg_compression":
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(p)])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else bgr
    elif kind == "noise":
        out = np.clip(bgr.astype(np.float32) + rng.normal(0, float(p), bgr.shape), 0, 255).astype(np.uint8)
    elif kind == "occlusion":
        out = bgr.copy()
        h, w = out.shape[:2]
        area = h * w * float(p)
        bw = min(w, max(8, int(math.sqrt(area) * rng.uniform(0.8, 1.4))))
        bh = min(h, max(8, int(area / max(bw, 1))))
        x0 = int(rng.integers(0, max(w - bw + 1, 1)))
        y0 = int(rng.integers(0, max(h - bh + 1, 1)))
        out[y0:y0 + bh, x0:x0 + bw] = 255 if rng.random() < 0.7 else 0
    elif kind == "color_cast":
        gains = np.ones(3, dtype=np.float32)
        ch = int(rng.integers(0, 3))
        gains[ch] = 1.0 + float(p) * rng.choice([-1.0, 1.0])
        out = np.clip(bgr.astype(np.float32) * gains[None, None, :], 0, 255).astype(np.uint8)
    else:
        raise ValueError(kind)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def iqa_features(img: Image.Image):
    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    centered = gray - gray.mean()
    spec = np.abs(np.fft.rfft2(centered)) ** 2
    fy = np.fft.fftfreq(gray.shape[0])[:, None]
    fx = np.fft.rfftfreq(gray.shape[1])[None, :]
    high_freq_ratio = float(spec[np.sqrt(fy ** 2 + fx ** 2) > 0.12].sum() / (spec.sum() + 1e-9))
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(hist.sum(), 1)
    entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum())
    noise_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    hp = cv2.filter2D(gray, -1, noise_kernel)
    mad = np.median(np.abs(hp - np.median(hp)))
    noise_sigma = float(mad * 1.4826 / 6.0)
    return {
        "laplacian_var": float(lap.var()),
        "laplacian_log": float(np.log1p(lap.var())),
        "tenengrad": float((grad ** 2).mean()),
        "edge_density": float((grad > 40).mean()),
        "high_freq_ratio": high_freq_ratio,
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "entropy": entropy,
        "noise_sigma": noise_sigma,
        "clipped_fraction": float(((gray < 5) | (gray > 250)).mean()),
    }


def quality_score(f):
    sharp = np.clip(f["laplacian_log"] / 8.0, 0, 1)
    contrast = np.clip(f["contrast"] / 0.28, 0, 1)
    brightness = 1.0 - min(abs(f["brightness"] - 0.65) / 0.65, 1.0)
    noise = 1.0 - np.clip(f["noise_sigma"] / 12.0, 0, 1)
    clipped = 1.0 - np.clip(f["clipped_fraction"] / 0.65, 0, 1)
    return float(np.mean([sharp, contrast, brightness, noise, clipped]))


def predict_batch(model, tensors, device):
    x = torch.stack(tensors).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu()
    conf, pred = probs.max(1)
    top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1] if probs.shape[1] > 1 else conf
    return pred.numpy(), conf.numpy(), margin.numpy()


def run_benchmark(model, data_dir, records, labels, idx2cat, device, batch_size):
    conditions = [("clean", 0)]
    for kind in PARAMS:
        for sev in [1, 2, 3]:
            conditions.append((kind, sev))

    rows, pending_tensors, pending_rows = [], [], []

    def flush():
        if not pending_tensors:
            return
        preds, confs, margins = predict_batch(model, pending_tensors, device)
        for j, row_idx in enumerate(pending_rows):
            rows[row_idx]["pred"] = int(preds[j])
            rows[row_idx]["pred_category"] = idx2cat[int(preds[j])]
            rows[row_idx]["confidence"] = float(confs[j])
            rows[row_idx]["margin"] = float(margins[j])
            rows[row_idx]["correct"] = int(preds[j] == rows[row_idx]["label"])
        pending_tensors.clear()
        pending_rows.clear()

    for i, (rec, label) in enumerate(zip(records, labels)):
        img = Image.open(data_dir / rec["img_local_path"]).convert("RGB")
        for kind, sev in conditions:
            rng = np.random.default_rng(abs(hash((rec["img_local_path"], kind, sev))) % (2 ** 32))
            dimg = degrade(img, kind, sev, rng)
            feats = iqa_features(dimg)
            row = {
                "image_id": Path(rec["img_local_path"]).name,
                "label": int(label),
                "true_category": idx2cat[int(label)],
                "degradation": kind,
                "severity": int(sev),
                "quality_score": quality_score(feats),
                **feats,
            }
            rows.append(row)
            pending_rows.append(len(rows) - 1)
            pending_tensors.append(eval_transform(dimg))
            if len(pending_tensors) >= batch_size:
                flush()
        if (i + 1) % 100 == 0:
            print(f"processed {i + 1}/{len(records)} test images", flush=True)
    flush()
    return pd.DataFrame(rows)


def make_plots(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for deg, sub in df.groupby("degradation"):
        g = sub.groupby("severity")["correct"].mean()
        ax.plot(g.index, g.values, marker="o", label=deg)
    ax.set_title("Fixed model accuracy vs degradation severity")
    ax.set_xlabel("severity")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_degradation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for deg, sub in df.groupby("degradation"):
        g = sub.groupby("severity")["confidence"].mean()
        ax.plot(g.index, g.values, marker="o", label=deg)
    ax.set_title("Fixed model confidence vs degradation severity")
    ax.set_xlabel("severity")
    ax.set_ylabel("mean confidence")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_by_degradation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([df[df.correct == 1].quality_score, df[df.correct == 0].quality_score], tick_labels=["correct", "wrong"])
    ax.set_title("No-reference quality score vs correctness")
    ax.set_ylabel("quality score")
    fig.tight_layout()
    fig.savefig(out_dir / "quality_vs_correct.png", dpi=150)
    plt.close(fig)


def write_html(df, out_dir):
    grouped = df.groupby(["degradation", "severity"]).agg(
        accuracy=("correct", "mean"),
        macro_f1=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
        mean_quality_score=("quality_score", "mean"),
        n=("correct", "size"),
    ).reset_index()
    rows = []
    for _, r in grouped.iterrows():
        rows.append(f"""
        <tr>
          <td>{html.escape(str(r.degradation))}</td>
          <td>{int(r.severity)}</td>
          <td>{r.accuracy:.3f}</td>
          <td>{r.mean_confidence:.3f}</td>
          <td>{r.mean_quality_score:.3f}</td>
          <td>{int(r.n)}</td>
        </tr>
        """)
    harmful = df[df.degradation != "clean"].groupby("degradation")["correct"].mean().sort_values().head(5)
    harmful_items = "".join(f"<li>{html.escape(k)}: {v:.3f} accuracy</li>" for k, v in harmful.items())
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixed Model Degradation Benchmark</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;color:#17202a}}
header{{padding:28px 36px;border-bottom:1px solid #d7dbdd}}main{{padding:24px 36px 40px}}
h1{{margin:0 0 6px;font-size:24px}}.sub{{margin:0;color:#5d6d7e}}table{{border-collapse:collapse;width:100%;margin-top:14px}}
th,td{{border-bottom:1px solid #d7dbdd;text-align:left;padding:8px;font-size:14px}}th{{background:#f4f6f7}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin:18px 0}}.metric{{background:#f4f6f7;padding:12px;border-radius:6px}}
img{{max-width:100%;border:1px solid #d7dbdd;border-radius:6px}}
</style></head><body>
<header><h1>Fixed Model Degradation Benchmark</h1><p class="sub">Controlled image-quality stress test after Grad-CAM shortcut analysis and fix.</p></header>
<main>
<section class="grid">
<div class="metric"><b>Rows</b><br>{len(df)}</div>
<div class="metric"><b>Images</b><br>{df.image_id.nunique()}</div>
<div class="metric"><b>Overall accuracy</b><br>{df.correct.mean():.3f}</div>
</section>
<h2>Most harmful degradation types</h2><ul>{harmful_items}</ul>
<h2>Results by degradation and severity</h2>
<table><thead><tr><th>Degradation</th><th>Severity</th><th>Accuracy</th><th>Confidence</th><th>Quality score</th><th>N</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Plots</h2>
<div class="grid">
<div><img src="accuracy_by_degradation.png"><p>Accuracy drop shows performance robustness.</p></div>
<div><img src="confidence_by_degradation.png"><p>Confidence shows whether the model knows it is uncertain.</p></div>
<div><img src="quality_vs_correct.png"><p>Quality score vs correctness tests if IQA can predict failures.</p></div>
</div>
</main></body></html>"""
    path = out_dir / "degradation_summary.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results_fixed_model_degradation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Optional test-image limit for smoke runs.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    records, labels, cat2idx, idx2cat = load_records(args.data_dir)
    test_records, test_labels = make_test_split(records, labels)
    if args.limit:
        test_records, test_labels = test_records[:args.limit], test_labels[:args.limit]

    model = load_model(args.checkpoint, len(cat2idx), device)
    print(f"Device: {device}")
    print(f"Test images: {len(test_records)}")
    df = run_benchmark(model, args.data_dir, test_records, test_labels, idx2cat, device, args.batch_size)
    csv_path = args.out_dir / "degradation_results.csv"
    df.to_csv(csv_path, index=False)
    make_plots(df, args.out_dir)
    html_path = write_html(df, args.out_dir)

    clean = df[df.degradation == "clean"]
    summary = [
        f"Device: {device}",
        f"Rows: {len(df)}",
        f"Images: {df.image_id.nunique()}",
        f"Clean accuracy: {clean.correct.mean():.4f}",
        f"Clean macro F1: {f1_score(clean.label, clean.pred, average='macro', zero_division=0):.4f}",
        f"All degraded accuracy: {df.correct.mean():.4f}",
        "",
        "Accuracy by degradation:",
        df.groupby("degradation")["correct"].mean().sort_values().to_string(),
        "",
        f"CSV: {csv_path}",
        f"HTML: {html_path}",
    ]
    (args.out_dir / "SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
