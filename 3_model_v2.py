"""
Model V2 — Improvements over Model 1:
  1. Aspect-ratio-preserving resize (pad to square, then resize)
  2. BatchNorm in classifier head (deeper head)
  3. Test-Time Augmentation (TTA) at inference
  4. EfficientNet-B2 (larger backbone)
  5. Town split by subcategory (Police, Fire, Airport, etc.)
  + All previous: Focal Loss, RandAugment, CutMix, MixUp, label smoothing,
    sqrt class weights, cosine annealing, early stopping, AdamW
"""

import json, os, time, warnings, random, math
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, top_k_accuracy_score
)
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIG

BASE_DIR = "/home/test/barco"
OUTPUT_DIR = os.path.join(BASE_DIR, "model_v2_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMG_SIZE = 260       # EfficientNet-B2 native input size
BATCH_SIZE = 24      # Smaller batch for B2
NUM_EPOCHS = 20
LR = 3e-4
NUM_WORKERS = 0
PATIENCE = 5
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.3
MIXUP_PROB = 0.3
TTA_TRANSFORMS = 5   # Number of TTA augmentations at test time


# TOWN SUBCATEGORY SPLITTER

def split_town(subcategory):
    """Split Town into visually meaningful sub-groups."""
    sub = subcategory.lower()
    if 'police' in sub: return 'Town - Police'
    if 'fire' in sub: return 'Town - Fire'
    if 'airport' in sub: return 'Town - Airport'
    if 'hospital' in sub or 'rescue' in sub: return 'Town - Rescue'
    if 'space' in sub: return 'Town - Space'
    if 'race' in sub or 'stuntz' in sub: return 'Town - Racing'
    if 'coast guard' in sub: return 'Town - Coast Guard'
    if 'construction' in sub: return 'Town - Construction'
    if any(x in sub for x in ['arctic', 'jungle', 'volcano', 'ocean', 'deep sea', 'exploration']): return 'Town - Exploration'
    return 'Town - General'
# MERGE MAP (same as before + Town split)

MERGE_MAP = {
    'The LEGO NINJAGO Movie': 'NINJAGO',
    'The LEGO Movie': 'LEGO Movies', 'The LEGO Movie 2': 'LEGO Movies',
    'DC Super Hero Girls': 'Super Heroes', 'Spider-Man': 'Super Heroes', 'Batman I': 'Super Heroes',
    'Elves': 'Friends & Fantasy', 'Friends': 'Friends & Fantasy',
    'DUPLO': 'Preschool', 'Primo': 'Preschool', 'Belville': 'Preschool', 'Scala': 'Preschool',
    'For Juniors': 'Preschool', 'Homemaker': 'Preschool', 'Fabuland': 'Preschool',
    'Pirates of the Caribbean': 'Pirates',
    'Castle': 'Castle & Medieval', 'Vikings': 'Castle & Medieval', 'Ninja': 'Castle & Medieval',
    'NEXO KNIGHTS': 'Castle & Medieval',
    'Space': 'Space & Sci-Fi', 'Aquazone': 'Space & Sci-Fi', 'Alpha Team': 'Space & Sci-Fi',
    'Exo-Force': 'Space & Sci-Fi', 'Power Miners': 'Space & Sci-Fi', 'Rock Raiders': 'Space & Sci-Fi',
    'Agents': 'Space & Sci-Fi', 'Ultra Agents': 'Space & Sci-Fi', 'Atlantis': 'Space & Sci-Fi',
    'Adventurers': 'Adventure', 'Indiana Jones': 'Adventure', "Pharaoh's Quest": 'Adventure',
    'Dino Attack': 'Adventure', 'Dino': 'Adventure', 'Western': 'Adventure',
    'SPEED CHAMPIONS': 'Racing & Vehicles', 'Racers': 'Racing & Vehicles',
    'World Racers': 'Racing & Vehicles', 'Speed Racer': 'Racing & Vehicles', 'Cars': 'Racing & Vehicles',
    'LEGENDS OF CHIMA': 'Fantasy', 'DREAMZzz': 'Fantasy', 'Hidden Side': 'Fantasy',
    'Monster Fighters': 'Fantasy',
    'The Hobbit and The Lord of the Rings': 'Middle-Earth',
    'Teenage Mutant Ninja Turtles': 'Licensed Entertainment',
    'SpongeBob SquarePants': 'Licensed Entertainment', 'Scooby-Doo': 'Licensed Entertainment',
    'The Simpsons': 'Licensed Entertainment', 'Ghostbusters': 'Licensed Entertainment',
    'Stranger Things': 'Licensed Entertainment', 'The Angry Birds Movie': 'Licensed Entertainment',
    'Trolls World Tour': 'Licensed Entertainment', 'The Incredibles': 'Licensed Entertainment',
    'Toy Story': 'Licensed Entertainment', 'Wednesday': 'Licensed Entertainment',
    'Wicked': 'Licensed Entertainment', 'Despicable Me and Minions': 'Licensed Entertainment',
    'The Lone Ranger': 'Licensed Entertainment', 'Prince of Persia': 'Licensed Entertainment',
    'Island Xtreme Stunts': 'Licensed Entertainment', 'The Powerpuff Girls': 'Licensed Entertainment',
    'Back to the Future': 'Licensed Entertainment', 'Friends TV Series': 'Licensed Entertainment',
    'Queer Eye': 'Licensed Entertainment', 'Lightyear': 'Licensed Entertainment',
    'Bluey': 'Licensed Entertainment', "Gabby's Dollhouse": 'Licensed Entertainment',
    'Dune': 'Licensed Entertainment', 'Star Trek': 'Licensed Entertainment',
    'Overwatch': 'Video Games', 'Fortnite': 'Video Games', 'Sonic the Hedgehog': 'Video Games',
    'Horizon': 'Video Games', 'The Legend of Zelda': 'Video Games', 'Animal Crossing': 'Video Games',
    'Avatar The Last Airbender': 'Video Games', 'Avatar': 'Video Games', 'One Piece': 'Video Games',
    'Collectible Minifigures': 'Collectible & Special', 'Holiday & Event': 'Collectible & Special',
    'Dimensions': 'Collectible & Special', 'Vidiyo': 'Collectible & Special',
    'Games': 'Collectible & Special',
    'LEGO Ideas': 'LEGO Promotional', 'BrickLink Designer Program': 'LEGO Promotional',
    'LEGO Brand': 'LEGO Promotional', 'LEGOLAND': 'LEGO Promotional',
    'LEGOLAND Parks': 'LEGO Promotional', 'Promotional': 'LEGO Promotional',
    'Studios': 'LEGO Promotional', 'FIRST LEGO League': 'LEGO Promotional',
    'Educational & Dacta': 'Education & Technical', 'BIONICLE': 'Education & Technical',
    'Hero Factory': 'Education & Technical', 'Technic': 'Education & Technical',
    'Basic': 'Education & Technical', 'FreeStyle': 'Education & Technical',
    'Master Builder Academy': 'Education & Technical', 'Building Bigger Thinking': 'Education & Technical',
    'Discovery': 'Education & Technical', 'Quatro': 'Education & Technical',
    'Universe': 'Education & Technical', 'Fusion': 'Education & Technical',
    'Nike': 'Education & Technical', 'Architecture': 'Education & Technical',
    'Clikits': 'Education & Technical', 'Unikitty!': 'Education & Technical',
}
# FOCAL LOSS

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            focal_weight = (1 - probs).pow(self.gamma)
            loss = -(focal_weight * smooth_targets * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt).pow(self.gamma)
            loss = focal_weight * ce_loss

        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w
        return loss.mean()


# CUTMIX & MIXUP

def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    return y1, y2, x1, x2

def apply_cutmix(images, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(images.device)
    y1, y2, x1, x2 = rand_bbox(images.size(), lam)
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (images.size(-1) * images.size(-2)))
    return images, targets, targets[index], lam

def apply_mixup(images, targets, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, targets, targets[index], lam


# ASPECT-RATIO-PRESERVING RESIZE

class PadToSquare:
    """Pad image to square with white background, preserving aspect ratio."""
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        # Pad with white (common LEGO image background)
        padded = ImageOps.pad(img, (max_side, max_side), color=(255, 255, 255))
        return padded


# LOAD DATA

print("Loading data...")
with open(os.path.join(BASE_DIR, "minifigs.json")) as f:
    data = json.load(f)

valid_data = [d for d in data if d.get('img_local_path') and
              os.path.exists(os.path.join(BASE_DIR, d['img_local_path']))]

# Apply merge + Town split
for d in valid_data:
    if d['category'] == 'Town':
        d['merged_category'] = split_town(d['subcategory'])
    else:
        d['merged_category'] = MERGE_MAP.get(d['category'], d['category'])

categories = [d['merged_category'] for d in valid_data]
cat_counts = Counter(categories)
cat_names = sorted(cat_counts.keys())
cat2idx = {c: i for i, c in enumerate(cat_names)}
idx2cat = {i: c for c, i in cat2idx.items()}
num_classes = len(cat_names)

print(f"Valid images: {len(valid_data)}, Classes: {num_classes}")
print("\nClass distribution:")
for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<35} {v:>5} ({v/len(valid_data)*100:.1f}%)")


# STRATIFIED SPLIT

labels = [cat2idx[d['merged_category']] for d in valid_data]
label_counts = Counter(labels)
small_idx = [i for i, l in enumerate(labels) if label_counts[l] < 7]
big_idx = [i for i, l in enumerate(labels) if label_counts[l] >= 7]
big_data = [valid_data[i] for i in big_idx]
big_labels = [labels[i] for i in big_idx]

train_data_big, temp_data, train_labels_big, temp_labels = train_test_split(
    big_data, big_labels, test_size=0.3, random_state=42, stratify=big_labels)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
train_data = train_data_big + [valid_data[i] for i in small_idx]
train_labels = train_labels_big + [labels[i] for i in small_idx]
print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


# DATASET

class MinifigDataset(Dataset):
    def __init__(self, records, labels, transform):
        self.records, self.labels, self.transform = records, labels, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        try: img = Image.open(os.path.join(BASE_DIR, self.records[idx]['img_local_path'])).convert('RGB')
        except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        return self.transform(img), self.labels[idx]

# Aspect-ratio-preserving transforms
train_transform = transforms.Compose([
    PadToSquare(),                                         # Pad to square first
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    PadToSquare(),                                         # Pad to square first
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# TTA transforms (multiple augmented views for test-time)
tta_transforms = [
    eval_transform,  # Original
    transforms.Compose([  # Horizontal flip
        PadToSquare(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([  # Slight rotation
        PadToSquare(),
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([  # Slight zoom (larger resize + center crop)
        PadToSquare(),
        transforms.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([  # Color jitter
        PadToSquare(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
]

train_dataset = MinifigDataset(train_data, train_labels, train_transform)
val_dataset = MinifigDataset(val_data, val_labels, eval_transform)
test_dataset = MinifigDataset(test_data, test_labels, eval_transform)

# Class weights
train_label_counts = Counter(train_labels)
class_weights = torch.zeros(num_classes)
for i in range(num_classes):
    count = train_label_counts.get(i, 1)
    class_weights[i] = 1.0 / math.sqrt(count)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(DEVICE)

sample_weights = [1.0 / math.sqrt(train_label_counts[l]) for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# MODEL: EfficientNet-B2 + deeper classifier head with BatchNorm

print("\nBuilding EfficientNet-B2 model...")
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

# Freeze first 4 blocks
for param in model.features[:4].parameters():
    param.requires_grad = False

# Deeper classifier head with BatchNorm
in_features = model.classifier[1].in_features  # 1408 for B2
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes)
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)")

criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
criterion_val = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# TRAINING

print(f"\nTraining for up to {NUM_EPOCHS} epochs...")
print(f"V2 improvements: EfficientNet-B2, BatchNorm head, aspect-ratio resize, Town split, TTA")
print("-" * 80)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
best_val_f1 = 0  # Early stop on macro F1 instead of accuracy
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for images, lbl in train_loader:
        images, lbl = images.to(DEVICE), lbl.to(DEVICE)

        r = random.random()
        if r < CUTMIX_PROB:
            images, targets_a, targets_b, lam = apply_cutmix(images, lbl, CUTMIX_ALPHA)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif r < CUTMIX_PROB + MIXUP_PROB:
            images, targets_a, targets_b, lam = apply_mixup(images, lbl, MIXUP_ALPHA)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, lbl)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        train_correct += preds.eq(lbl).sum().item()
        train_total += images.size(0)

    scheduler.step(epoch)
    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_preds_all, val_labels_all = [], []
    with torch.no_grad():
        for images, lbl in val_loader:
            images, lbl = images.to(DEVICE), lbl.to(DEVICE)
            outputs = model(images)
            loss = criterion_val(outputs, lbl)
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_correct += preds.eq(lbl).sum().item()
            val_total += images.size(0)
            val_preds_all.extend(preds.cpu().numpy())
            val_labels_all.extend(lbl.cpu().numpy())

    val_loss /= val_total
    val_acc = val_correct / val_total
    val_macro_f1 = f1_score(val_labels_all, val_preds_all, average='macro', zero_division=0)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_macro_f1)

    lr = optimizer.param_groups[0]['lr']
    elapsed = time.time() - t0
    marker = ""
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        patience_counter = 0
        marker = " * best"
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_macro_f1:.4f} | "
          f"LR: {lr:.6f} | {elapsed:.0f}s{marker}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nBest validation macro F1: {best_val_f1:.4f}")

# TEST-TIME AUGMENTATION (TTA) EVALUATION

print("\n" + "=" * 80)
print("EVALUATING ON TEST SET WITH TTA ({} augmented views)".format(len(tta_transforms)))
print("=" * 80)

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE, weights_only=True))
model.eval()

# Standard evaluation (no TTA) first
all_preds_no_tta, all_labels_arr, all_probs_no_tta = [], [], []
with torch.no_grad():
    for images, lbl in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, preds = outputs.max(1)
        all_preds_no_tta.extend(preds.cpu().numpy())
        all_labels_arr.extend(lbl.numpy())
        all_probs_no_tta.extend(probs)

all_labels_arr = np.array(all_labels_arr)
all_probs_no_tta = np.array(all_probs_no_tta)
all_preds_no_tta = np.array(all_preds_no_tta)

# TTA evaluation
print("Running TTA...")
all_probs_tta = np.zeros_like(all_probs_no_tta)

for t_idx, tta_t in enumerate(tta_transforms):
    tta_dataset = MinifigDataset(test_data, test_labels, tta_t)
    tta_loader = DataLoader(tta_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    probs_this = []
    with torch.no_grad():
        for images, _ in tta_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            probs_this.extend(probs)

    all_probs_tta += np.array(probs_this)
    print(f"  TTA view {t_idx+1}/{len(tta_transforms)} done")

all_probs_tta /= len(tta_transforms)
all_preds_tta = all_probs_tta.argmax(axis=1)


# METRICS — both with and without TTA

def compute_metrics(preds, labels, probs, label=''):
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    macro_prec = precision_score(labels, preds, average='macro', zero_division=0)
    macro_rec = recall_score(labels, preds, average='macro', zero_division=0)
    top3 = top_k_accuracy_score(labels, probs, k=3, labels=range(num_classes))
    top5 = top_k_accuracy_score(labels, probs, k=5, labels=range(num_classes))
    return {'Accuracy': acc, 'Top-3 Accuracy': top3, 'Top-5 Accuracy': top5,
            'Macro F1': macro_f1, 'Weighted F1': weighted_f1,
            'Macro Precision': macro_prec, 'Macro Recall': macro_rec}

metrics_no_tta = compute_metrics(all_preds_no_tta, all_labels_arr, all_probs_no_tta)
metrics_tta = compute_metrics(all_preds_tta, all_labels_arr, all_probs_tta)

print(f"\n{'Metric':<30} {'No TTA':>10} {'With TTA':>10} {'TTA Δ':>8}")
print("-" * 60)
for metric in metrics_no_tta:
    v1, v2 = metrics_no_tta[metric], metrics_tta[metric]
    delta = v2 - v1
    arrow = "+" if delta > 0 else ""
    print(f"{metric:<30} {v1:>10.4f} {v2:>10.4f} {arrow}{delta:>7.4f}")

# Use TTA results for final evaluation
all_preds = all_preds_tta
all_probs = all_probs_tta

present_labels = sorted(set(all_labels_arr) | set(all_preds))
present_names = [idx2cat[i] for i in present_labels]

report = classification_report(all_labels_arr, all_preds, labels=present_labels,
                                target_names=present_names, zero_division=0, output_dict=True)
report_str = classification_report(all_labels_arr, all_preds, labels=present_labels,
                                    target_names=present_names, zero_division=0)
print("\n" + report_str)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
    f.write("V2 Results (with TTA)\n")
    for k, v in metrics_tta.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write(f"\n{report_str}")

# PLOTS


# 1. Training curves (now includes val F1)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
ax2.plot(history['train_acc'], label='Train'); ax2.plot(history['val_acc'], label='Val')
ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
ax3.plot(history['val_f1'], label='Val Macro F1', color='green')
ax3.set_title('Validation Macro F1'); ax3.set_xlabel('Epoch'); ax3.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150); plt.close()

# 2. Confusion matrix
cm = confusion_matrix(all_labels_arr, all_preds, labels=present_labels)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
fig, ax = plt.subplots(figsize=(26, 22))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=present_names, yticklabels=present_names, ax=ax, vmin=0, vmax=1)
ax.set_title('Normalized Confusion Matrix — V2 (with TTA)')
ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.xticks(rotation=45, ha='right', fontsize=7); plt.yticks(rotation=0, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150); plt.close()

# 3. Per-class F1
per_class_f1 = [(cat, report[cat]['f1-score'], report[cat]['support']) for cat in present_names]
per_class_f1.sort(key=lambda x: -x[1])
fig, ax = plt.subplots(figsize=(14, 14))
names_p = [x[0] for x in per_class_f1]
f1s = [x[1] for x in per_class_f1]
sups = [x[2] for x in per_class_f1]
colors = ['green' if f > 0.5 else 'orange' if f > 0.2 else 'red' for f in f1s]
ax.barh(range(len(names_p)), f1s, color=colors)
ax.set_yticks(range(len(names_p)))
ax.set_yticklabels([f"{n} (n={s})" for n, s in zip(names_p, sups)], fontsize=8)
ax.set_xlabel('F1 Score'); ax.set_title('Per-Class F1 — V2 (with TTA)')
ax.invert_yaxis(); ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1.png"), dpi=150); plt.close()

# 4. Confused pairs
print(f"\n{'='*80}")
print("TOP 15 MOST CONFUSED PAIRS")
print(f"{'='*80}")
full_cm = confusion_matrix(all_labels_arr, all_preds, labels=range(num_classes))
confused = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and full_cm[i][j] > 0:
            confused.append((idx2cat[i], idx2cat[j], full_cm[i][j]))
confused.sort(key=lambda x: -x[2])
print(f"{'True':<35} {'Predicted As':<35} {'Count':>5}")
print("-" * 77)
for t, p, c in confused[:15]:
    print(f"{t:<35} {p:<35} {c:>5}")

# 5. Zero F1
zero_f1 = [(n, s) for n, f, s in per_class_f1 if f == 0]
print(f"\nCategories with F1=0: {len(zero_f1)}")
for n, s in zero_f1: print(f"  {n} (test: {s})")

# 6. Size buckets
print(f"\n{'='*80}")
print("PERFORMANCE BY CATEGORY SIZE")
print(f"{'='*80}")
test_counts = Counter(all_labels_arr)
buckets = {'<10': [], '10-49': [], '50-99': [], '100-499': [], '500+': []}
for cat in present_names:
    idx = cat2idx[cat]
    support = test_counts.get(idx, 0)
    f1 = report[cat]['f1-score']
    if support < 10: buckets['<10'].append(f1)
    elif support < 50: buckets['10-49'].append(f1)
    elif support < 100: buckets['50-99'].append(f1)
    elif support < 500: buckets['100-499'].append(f1)
    else: buckets['500+'].append(f1)

print(f"{'Bucket':<15} {'#Classes':>10} {'Avg F1':>10} {'Min F1':>10} {'Max F1':>10}")
print("-" * 57)
for bucket, f1s_list in buckets.items():
    if f1s_list:
        print(f"{bucket:<15} {len(f1s_list):>10} {np.mean(f1s_list):>10.4f} {min(f1s_list):>10.4f} {max(f1s_list):>10.4f}")

print(f"\nAll results saved to: {OUTPUT_DIR}")
