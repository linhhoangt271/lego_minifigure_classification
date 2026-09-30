"""Create a self-contained HTML Grad-CAM demo for one selected image.

This is an inference/explanation demo, not a live training visualizer. It shows
the trained V3 model's classification flow for one image:

raw image -> preprocessing -> prediction probabilities -> Grad-CAM heatmap

Example:
    python3 v3_gradcam_html_demo.py \
      --data-dir "/home/yourname/lego_project/Assignment 2_final" \
      --checkpoint "/home/yourname/lego_project/Assignment 2_final/optionB_v3_results/best_model_focal.pth" \
      --variant focal \
      --image-id SW1321.jpg \
      --out-html demo_gradcam.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
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
        arcface = ArcFaceHead(512, num_classes)
        return model, arcface
    classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return nn.Sequential(backbone, classifier), None


def load_v3(checkpoint_path: Path, variant: str, num_classes: int, device):
    model, arcface = build_v3_model(variant, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        if variant == "arcface":
            arcface.load_state_dict(checkpoint["arcface_head"])
    else:
        model.load_state_dict(checkpoint)
    if variant == "arcface":
        wrapped = ArcFaceInference(model, arcface).to(device).eval()
        return wrapped, model[0]
    return model.to(device).eval(), model[0]


class GradCAM:
    def __init__(self, model, target_layer):
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
        logits[:, target_class].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam, probs[0].detach().cpu().numpy()


def unnormalize(x):
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(STD) + np.array(MEAN)
    return np.clip(arr, 0, 1)


def overlay_cam(rgb, cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.55 * rgb + 0.45 * heat, 0, 1)


def image_to_base64(img_array_or_pil) -> str:
    if isinstance(img_array_or_pil, Image.Image):
        img = img_array_or_pil.convert("RGB")
    else:
        arr = np.asarray(img_array_or_pil)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def find_record(records, labels, data_dir: Path, image_id: str | None):
    if image_id is None:
        test_records, test_labels = make_test_split(records, labels)
        return test_records[0], test_labels[0]
    for rec, label in zip(records, labels):
        if Path(rec["img_local_path"]).name == image_id or rec["img_local_path"] == image_id:
            return rec, label
    raise ValueError(f"Could not find image-id {image_id!r} under {data_dir}")


def probability_bars(probs, idx2cat, top_k):
    order = np.argsort(-probs)[:top_k]
    rows = []
    for rank, idx in enumerate(order, 1):
        pct = float(probs[idx] * 100)
        rows.append(
            f"""
            <div class="prob-row">
              <div class="prob-label"><span>{rank}.</span> {html.escape(idx2cat[int(idx)])}</div>
              <div class="bar"><div style="width:{pct:.2f}%"></div></div>
              <div class="prob-num">{pct:.2f}%</div>
            </div>
            """
        )
    return "\n".join(rows)


def write_html(out_path: Path, context: dict):
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V3 Grad-CAM Classification Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6d7e;
      --line: #d7dbdd;
      --soft: #f4f6f7;
      --accent: #21618c;
      --good: #1e8449;
      --bad: #b03a2e;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 28px 36px 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    main {{
      padding: 24px 36px 40px;
      display: grid;
      gap: 24px;
    }}
    section {{
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 17px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .metric {{
      background: var(--soft);
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 14px;
    }}
    .metric b {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
      margin-bottom: 3px;
    }}
    .images {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    figure {{
      margin: 0;
    }}
    figure img {{
      width: 100%;
      max-height: 460px;
      object-fit: contain;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      counter-reset: step;
    }}
    .step {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      min-height: 86px;
    }}
    .step:before {{
      counter-increment: step;
      content: counter(step);
      display: inline-grid;
      place-items: center;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: var(--accent);
      color: white;
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .step b {{
      display: block;
      margin-bottom: 4px;
    }}
    .step span {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }}
    .prob-row {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(160px, 3fr) 72px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      font-size: 14px;
    }}
    .prob-label span {{
      color: var(--muted);
    }}
    .bar {{
      height: 12px;
      background: #e5e8e8;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar div {{
      height: 100%;
      background: var(--accent);
    }}
    .prob-num {{
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .status {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      color: white;
      background: {context["status_color"]};
    }}
    code {{
      background: var(--soft);
      padding: 1px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>V3 Grad-CAM Classification Demo</h1>
    <p class="sub">Single-image explanation of the trained LEGO classifier before IQA/degradation testing.</p>
  </header>
  <main>
    <section>
      <h2>Selected Image</h2>
      <div class="meta">
        <div class="metric"><b>Image</b>{html.escape(context["image_id"])}</div>
        <div class="metric"><b>True Class</b>{html.escape(context["true_category"])}</div>
        <div class="metric"><b>Predicted Class</b>{html.escape(context["pred_category"])}</div>
        <div class="metric"><b>Confidence</b>{context["confidence"]:.2%}</div>
        <div class="metric"><b>Result</b><span class="status">{context["status"]}</span></div>
      </div>
    </section>

    <section>
      <h2>Classification Process</h2>
      <div class="steps">
        <div class="step"><b>Input image</b><span>The selected BrickLink image is loaded from disk.</span></div>
        <div class="step"><b>Preprocessing</b><span>Image is padded to square, resized to <code>380x380</code>, converted to tensor, and ImageNet-normalized.</span></div>
        <div class="step"><b>Forward pass</b><span>EfficientNet-B4 extracts visual features and the classifier head predicts class probabilities.</span></div>
        <div class="step"><b>Grad-CAM</b><span>Gradients from the predicted class are backpropagated to the final convolution layer to locate influential regions.</span></div>
      </div>
    </section>

    <section>
      <h2>What The Model Looked At</h2>
      <div class="images">
        <figure>
          <img src="data:image/png;base64,{context["original_b64"]}" alt="Original image">
          <figcaption>Original preprocessed image.</figcaption>
        </figure>
        <figure>
          <img src="data:image/png;base64,{context["heatmap_b64"]}" alt="Grad-CAM heatmap">
          <figcaption>Grad-CAM heatmap. Red/yellow regions contributed most to the prediction.</figcaption>
        </figure>
        <figure>
          <img src="data:image/png;base64,{context["overlay_b64"]}" alt="Grad-CAM overlay">
          <figcaption>Overlay of the heatmap on the image.</figcaption>
        </figure>
      </div>
    </section>

    <section>
      <h2>Top Predictions</h2>
      {context["probability_bars"]}
    </section>

    <section>
      <h2>How To Read This</h2>
      <p>
        A good explanation should highlight meaningful minifigure regions such as the head,
        torso print, helmet, clothing, or accessories. If the strongest heatmap regions are
        mostly background, image border, or irrelevant catalogue artifacts, that suggests
        shortcut learning and should be investigated before the IQA robustness study.
      </p>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with minifigs.json and images/.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="V3 checkpoint path.")
    parser.add_argument("--variant", choices=["focal", "arcface", "supcon"], default="focal")
    parser.add_argument("--image-id", type=str, default=None, help="Filename like SW1321.jpg. Defaults to first test image.")
    parser.add_argument("--out-html", type=Path, default=Path("v3_gradcam_single_image_demo.html"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    records, labels, _, idx2cat = load_records(args.data_dir)
    rec, true_label = find_record(records, labels, args.data_dir, args.image_id)
    image_path = args.data_dir / rec["img_local_path"]
    image = Image.open(image_path).convert("RGB")

    model, backbone = load_v3(args.checkpoint, args.variant, len(idx2cat), device)
    cammer = GradCAM(model, backbone.conv_head)

    x = eval_transform(image).unsqueeze(0).to(device)
    cam, probs = cammer(x)
    cammer.remove()

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    rgb = unnormalize(x[0])
    overlay = overlay_cam(rgb, cam)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    correct = pred == true_label
    context = {
        "image_id": Path(rec["img_local_path"]).name,
        "true_category": idx2cat[int(true_label)],
        "pred_category": idx2cat[pred],
        "confidence": confidence,
        "status": "correct" if correct else "wrong",
        "status_color": "#1e8449" if correct else "#b03a2e",
        "original_b64": image_to_base64(rgb),
        "heatmap_b64": image_to_base64(heat),
        "overlay_b64": image_to_base64(overlay),
        "probability_bars": probability_bars(probs, idx2cat, args.top_k),
    }
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    write_html(args.out_html, context)
    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print(f"True: {context['true_category']}")
    print(f"Pred: {context['pred_category']} ({confidence:.2%})")
    print(f"Wrote HTML: {args.out_html}")


if __name__ == "__main__":
    main()
