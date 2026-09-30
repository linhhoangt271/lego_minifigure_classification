"""
Baseline Model: Image-based LEGO Minifigure Category Classification
- Transfer learning with EfficientNet-B0
- Raw 122 categories (no merging)
- Stratified train/val/test split
- Full evaluation metrics
"""

import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, top_k_accuracy_score
)
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


### CONFIG 

BASE_DIR = "/home/test/barco"
JSON_PATH = os.path.join(BASE_DIR, "minifigs.json")
IMG_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "baseline_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")
print(f"Using device: {DEVICE}")

IMG_SIZE = 128  # Smaller for faster baseline
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 0  # MPS compatibility


### LOAD DATA

print("Loading data...")
with open(JSON_PATH) as f:
    data = json.load(f)

# Filter to records with existing images
valid_data = []
for d in data:
    if d.get('img_local_path'):
        img_path = os.path.join(BASE_DIR, d['img_local_path'])
        if os.path.exists(img_path):
            valid_data.append(d)

print(f"Total records: {len(data)}, Valid images: {len(valid_data)}")

# Build label encoding
categories = [d['category'] for d in valid_data]
cat_counts = Counter(categories)
cat_names = sorted(cat_counts.keys())
cat2idx = {c: i for i, c in enumerate(cat_names)}
idx2cat = {i: c for c, i in cat2idx.items()}
num_classes = len(cat_names)
print(f"Number of classes: {num_classes}")


# STRATIFIED SPLIT (handle tiny classes)

labels = [cat2idx[d['category']] for d in valid_data]

# Categories with <7 samples can't reliably stratify into 3 splits
# Put them all in train
label_counts = Counter(labels)
small_idx = [i for i, l in enumerate(labels) if label_counts[l] < 7]
big_idx = [i for i, l in enumerate(labels) if label_counts[l] >= 7]

big_data = [valid_data[i] for i in big_idx]
big_labels = [labels[i] for i in big_idx]

train_data_big, temp_data, train_labels_big, temp_labels = train_test_split(
    big_data, big_labels, test_size=0.3, random_state=42, stratify=big_labels
)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# Add tiny-class samples to train
train_data = train_data_big + [valid_data[i] for i in small_idx]
train_labels = train_labels_big + [labels[i] for i in small_idx]

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


# DATASET

class MinifigDataset(Dataset):
    def __init__(self, records, labels, transform=None):
        self.records = records
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path = os.path.join(BASE_DIR, self.records[idx]['img_local_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = MinifigDataset(train_data, train_labels, train_transform)
val_dataset = MinifigDataset(val_data, val_labels, eval_transform)
test_dataset = MinifigDataset(test_data, test_labels, eval_transform)


# CLASS WEIGHTS + SAMPLER

train_label_counts = Counter(train_labels)
class_weights = torch.zeros(num_classes)
for i in range(num_classes):
    count = train_label_counts.get(i, 1)
    class_weights[i] = 1.0 / count
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(DEVICE)

# Weighted sampler for balanced batches
sample_weights = [1.0 / train_label_counts[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# MODEL: EfficientNet-B0 (pretrained)

print("Building model...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze early layers, fine-tune last blocks
for param in model.features[:6].parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)


# TRAINING LOOP

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print("-" * 65)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    t0 = time.time()

    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for images, labels_batch in train_loader:
        images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        train_correct += preds.eq(labels_batch).sum().item()
        train_total += images.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels_batch)

            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels_batch).sum().item()
            val_total += images.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    elapsed = time.time() - t0
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {lr:.6f} | {elapsed:.0f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

print(f"\nBest validation accuracy: {best_val_acc:.4f}")


# EVALUATION ON TEST SET

print("\n" + "=" * 65)
print("EVALUATING ON TEST SET")
print("=" * 65)

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), weights_only=True))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.numpy())
        all_probs.extend(probs)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# METRICS

# Overall metrics
acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

# Top-k accuracy
top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3, labels=range(num_classes))
top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5, labels=range(num_classes))

print(f"\n{'Metric':<30} {'Value':>10}")
print("-" * 42)
print(f"{'Accuracy':<30} {acc:>10.4f}")
print(f"{'Top-3 Accuracy':<30} {top3_acc:>10.4f}")
print(f"{'Top-5 Accuracy':<30} {top5_acc:>10.4f}")
print(f"{'Macro F1':<30} {macro_f1:>10.4f}")
print(f"{'Weighted F1':<30} {weighted_f1:>10.4f}")
print(f"{'Macro Precision':<30} {macro_prec:>10.4f}")
print(f"{'Macro Recall':<30} {macro_rec:>10.4f}")

# Per-class report
# Use explicit labels to handle missing classes in test set
present_labels = sorted(set(all_labels) | set(all_preds))
present_names = [idx2cat[i] for i in present_labels]

report = classification_report(
    all_labels, all_preds,
    labels=present_labels,
    target_names=present_names,
    zero_division=0,
    output_dict=True
)
report_str = classification_report(
    all_labels, all_preds,
    labels=present_labels,
    target_names=present_names,
    zero_division=0
)
print("\n" + report_str)

# Save report to file
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n")
    f.write(f"Top-5 Accuracy: {top5_acc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write(report_str)


# PLOTS
#

# 1. Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Val')
ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'], label='Val')
ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close()

# 2. Confusion matrix (top-20 categories for readability)
top20_cats = [c for c, _ in Counter(all_labels).most_common(20)]
mask = np.isin(all_labels, top20_cats)
filtered_labels = all_labels[mask]
filtered_preds = all_preds[mask]

top20_names = [idx2cat[i] for i in top20_cats]
cm = confusion_matrix(filtered_labels, filtered_preds, labels=top20_cats)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=top20_names, yticklabels=top20_names, ax=ax,
            vmin=0, vmax=1)
ax.set_title('Normalized Confusion Matrix (Top 20 Categories)')
ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_top20.png"), dpi=150)
plt.close()

# 3. Per-class F1 bar chart
per_class_f1 = []
for cat in present_names:
    per_class_f1.append((cat, report[cat]['f1-score'], report[cat]['support']))

per_class_f1.sort(key=lambda x: -x[1])

fig, ax = plt.subplots(figsize=(16, 20))
names = [x[0] for x in per_class_f1]
f1s = [x[1] for x in per_class_f1]
supports = [x[2] for x in per_class_f1]
colors = ['green' if f > 0.5 else 'orange' if f > 0.2 else 'red' for f in f1s]
bars = ax.barh(range(len(names)), f1s, color=colors)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([f"{n} (n={s})" for n, s in zip(names, supports)], fontsize=7)
ax.set_xlabel('F1 Score')
ax.set_title('Per-Class F1 Score (green >0.5, orange >0.2, red ≤0.2)')
ax.invert_yaxis()
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1.png"), dpi=150)
plt.close()

# 4. Most confused pairs
print("\n" + "=" * 65)
print("TOP 20 MOST CONFUSED CATEGORY PAIRS")
print("=" * 65)

full_cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
confused_pairs = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and full_cm[i][j] > 0:
            confused_pairs.append((idx2cat[i], idx2cat[j], full_cm[i][j]))

confused_pairs.sort(key=lambda x: -x[2])
print(f"{'True Category':<40} {'Predicted As':<40} {'Count':>5}")
print("-" * 87)
for true_c, pred_c, count in confused_pairs[:20]:
    print(f"{true_c:<40} {pred_c:<40} {count:>5}")

# 5. Categories with 0% F1
print("\n" + "=" * 65)
print("CATEGORIES WITH F1 = 0 (completely failed)")
print("=" * 65)
zero_f1 = [(n, s) for n, f, s in per_class_f1 if f == 0]
for n, s in zero_f1:
    print(f"  {n} (test samples: {s})")
print(f"\nTotal: {len(zero_f1)} categories with F1=0")

# 6. Summary by size bucket
print("\n" + "=" * 65)
print("PERFORMANCE BY CATEGORY SIZE")
print("=" * 65)
test_counts = Counter(all_labels)
buckets = {'<10': [], '10-49': [], '50-99': [], '100-299': [], '300+': []}
for cat in present_names:
    idx = cat2idx[cat]
    support = test_counts.get(idx, 0)
    f1 = report[cat]['f1-score']
    if support < 10: buckets['<10'].append(f1)
    elif support < 50: buckets['10-49'].append(f1)
    elif support < 100: buckets['50-99'].append(f1)
    elif support < 300: buckets['100-299'].append(f1)
    else: buckets['300+'].append(f1)

print(f"{'Size Bucket':<15} {'#Classes':>10} {'Avg F1':>10} {'Min F1':>10} {'Max F1':>10}")
print("-" * 57)
for bucket, f1s_list in buckets.items():
    if f1s_list:
        print(f"{bucket:<15} {len(f1s_list):>10} {np.mean(f1s_list):>10.4f} {min(f1s_list):>10.4f} {max(f1s_list):>10.4f}")

print(f"\nAll results saved to: {OUTPUT_DIR}")
