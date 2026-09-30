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
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from v3_gradcam_demo import (
    eval_transform,
    load_records,
    load_v3,
    make_test_split,
    overlay_cam,
    unnormalize,
    GradCAM,
)


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
