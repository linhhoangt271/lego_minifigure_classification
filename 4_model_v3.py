"""
Model V3 — Improvements over V2:
  1. YOLO-based minifigure cropping + synthetic augmentation for minority classes
  2. EfficientNet-B4 backbone (380px native)
  3. Three loss function variants: Focal, ArcFace, SupCon+CE
  4. Pick best variant automatically
  + All previous: CutMix, MixUp, label smoothing, sqrt class weights,
    cosine annealing, early stopping, AdamW, TTA, PadToSquare, Town split
"""

import json, os, time, warnings, random, math, shutil, copy
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "model_v3_results")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                       else "cpu")
print(f"Using device: {DEVICE}")

IMG_SIZE = 380        # EfficientNet-B4 native input size
BATCH_SIZE = 16       # Larger model + images
NUM_EPOCHS = 20
LR = 3e-4
NUM_WORKERS = 4
PATIENCE = 5
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.3
MIXUP_PROB = 0.3
TTA_TRANSFORMS = 5
MIN_SAMPLES_TARGET = 50  # Target minimum training samples per class

# TOWN SUBCATEGORY SPLITTER

def split_town(subcategory):
    sub = subcategory.lower()
    if 'police' in sub: return 'Town - Police'
    if 'fire' in sub: return 'Town - Fire'
    if 'airport' in sub: return 'Town - Airport'
    if 'hospital' in sub or 'rescue' in sub: return 'Town - Rescue'
    if 'space' in sub: return 'Town - Space'
    if 'race' in sub or 'stuntz' in sub: return 'Town - Racing'
    if 'coast guard' in sub: return 'Town - Coast Guard'
    if 'construction' in sub: return 'Town - Construction'
    if any(x in sub for x in ['arctic', 'jungle', 'volcano', 'ocean', 'deep sea', 'exploration']):
        return 'Town - Exploration'
    return 'Town - General'


# MERGE MAP (same as V2)

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


# ARCFACE LOSS

class ArcFaceHead(nn.Module):
    """ArcFace angular margin classification head."""
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
        # Normalize
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale

class ArcFaceLoss(nn.Module):
    """CrossEntropy on ArcFace logits with optional label smoothing."""
    def __init__(self, weight=None, label_smoothing=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


# SUPERVISED CONTRASTIVE LOSS

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D) normalized embeddings
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.clamp(mask_pos_pairs, min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -mean_log_prob_pos.mean()
        return loss


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
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        padded = ImageOps.pad(img, (max_side, max_side), color=(255, 255, 255))
        return padded


# LOAD DATA

print("Loading data...")
with open(os.path.join(BASE_DIR, "minifigs.json")) as f:
    data = json.load(f)

valid_data = [d for d in data if d.get('img_local_path') and
              os.path.exists(os.path.join(BASE_DIR, d['img_local_path']))]

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


# STEP 2: YOLO CROP + SYNTHETIC AUGMENTATION FOR MINORITY CLASSES

print("\n" + "=" * 80)
print("YOLO CROP + SYNTHETIC AUGMENTATION FOR MINORITY CLASSES")
print("=" * 80)

import albumentations as A
import cv2

# Heavy augmentation pipeline for synthetic generation
heavy_aug = A.Compose([
    A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.8),
    A.Perspective(scale=(0.05, 0.12), p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.8),
    A.ElasticTransform(alpha=30, sigma=5, p=0.3),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])


def yolo_crop_image(yolo_model, img_path):
    """Use YOLO to detect and crop the main object from an image."""
    try:
        results = yolo_model(img_path, verbose=False, conf=0.25)
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get largest detection by area
            boxes = results[0].boxes
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best_idx = areas.argmax().item()
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            img = cv2.imread(img_path)
            if img is None:
                return None
            # Add small padding around crop
            h, w = img.shape[:2]
            pad = int(max(x2 - x1, y2 - y1) * 0.05)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                return crop
    except Exception:
        pass
    return None


def generate_synthetic_data(train_data, train_labels, cat2idx, idx2cat):
    """Generate synthetic augmented images for minority classes using YOLO crops."""
    from ultralytics import YOLO

    # Count training samples per class
    train_counts = Counter(train_labels)
    minority_classes = {cls_idx: count for cls_idx, count in train_counts.items()
                        if count < MIN_SAMPLES_TARGET}

    if not minority_classes:
        print("No minority classes found (all have >= {} samples)".format(MIN_SAMPLES_TARGET))
        return [], []

    print(f"Found {len(minority_classes)} minority classes (< {MIN_SAMPLES_TARGET} samples)")
    for cls_idx, count in sorted(minority_classes.items(), key=lambda x: x[1]):
        print(f"  {idx2cat[cls_idx]:<35} {count:>4} samples (need ~{MIN_SAMPLES_TARGET - count} more)")

    # Load YOLOv8 for object detection
    print("\nLoading YOLOv8 model...")
    yolo_model = YOLO("yolov8n.pt")  # Nano model, fast

    synthetic_records = []
    synthetic_labels = []
    generated_count = 0

    for cls_idx, count in sorted(minority_classes.items(), key=lambda x: x[1]):
        cls_name = idx2cat[cls_idx]
        # Sanitize class name for directory
        safe_name = cls_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
        cls_synth_dir = os.path.join(SYNTHETIC_DIR, safe_name)
        os.makedirs(cls_synth_dir, exist_ok=True)

        # Get all training images for this class
        cls_indices = [i for i, l in enumerate(train_labels) if l == cls_idx]
        cls_records = [train_data[i] for i in cls_indices]

        # How many synthetic images needed
        num_needed = MIN_SAMPLES_TARGET - count
        # Generate 5-10x augmented versions per image
        augs_per_image = max(5, min(10, math.ceil(num_needed / max(len(cls_records), 1))))

        cls_generated = 0
        for rec in cls_records:
            img_path = os.path.join(BASE_DIR, rec['img_local_path'])

            # Try YOLO crop first
            cropped = yolo_crop_image(yolo_model, img_path)
            if cropped is None:
                # Fallback: use original image
                cropped = cv2.imread(img_path)
                if cropped is None:
                    continue

            # Generate augmented versions
            for aug_i in range(augs_per_image):
                if cls_generated >= num_needed:
                    break
                augmented = heavy_aug(image=cropped)['image']
                fname = f"{safe_name}_{cls_generated:04d}.jpg"
                fpath = os.path.join(cls_synth_dir, fname)
                cv2.imwrite(fpath, augmented)

                # Create a synthetic record
                synth_rec = {'img_local_path': os.path.relpath(fpath, BASE_DIR),
                             'merged_category': cls_name,
                             '_synthetic': True}
                synthetic_records.append(synth_rec)
                synthetic_labels.append(cls_idx)
                cls_generated += 1

            if cls_generated >= num_needed:
                break

        generated_count += cls_generated
        print(f"  {cls_name:<35} +{cls_generated} synthetic → {count + cls_generated} total")

    print(f"\nTotal synthetic images generated: {generated_count}")
    return synthetic_records, synthetic_labels


synthetic_records, synthetic_labels = generate_synthetic_data(
    train_data, train_labels, cat2idx, idx2cat)

# Merge synthetic data into training set
train_data_full = train_data + synthetic_records
train_labels_full = train_labels + synthetic_labels
print(f"Training set: {len(train_data)} original + {len(synthetic_records)} synthetic = {len(train_data_full)} total")


# DATASET

class MinifigDataset(Dataset):
    def __init__(self, records, labels, transform):
        self.records, self.labels, self.transform = records, labels, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        try:
            img_path = self.records[idx]['img_local_path']
            if not os.path.isabs(img_path):
                img_path = os.path.join(BASE_DIR, img_path)
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        return self.transform(img), self.labels[idx]


class EmbeddingDataset(Dataset):
    """Returns (image, label) for contrastive learning — needs two views."""
    def __init__(self, records, labels, transform):
        self.records, self.labels, self.transform = records, labels, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        try:
            img_path = self.records[idx]['img_local_path']
            if not os.path.isabs(img_path):
                img_path = os.path.join(BASE_DIR, img_path)
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, self.labels[idx]


# Transforms
train_transform = transforms.Compose([
    PadToSquare(),
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
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tta_transforms = [
    eval_transform,
    transforms.Compose([
        PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        PadToSquare(), transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.CenterCrop(IMG_SIZE), transforms.RandomRotation(10),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        PadToSquare(), transforms.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
]

# Datasets
train_dataset = MinifigDataset(train_data_full, train_labels_full, train_transform)
val_dataset = MinifigDataset(val_data, val_labels, eval_transform)
test_dataset = MinifigDataset(test_data, test_labels, eval_transform)

# Class weights (based on full training set including synthetic)
train_label_counts = Counter(train_labels_full)
class_weights = torch.zeros(num_classes)
for i in range(num_classes):
    count = train_label_counts.get(i, 1)
    class_weights[i] = 1.0 / math.sqrt(count)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(DEVICE)

sample_weights = [1.0 / math.sqrt(train_label_counts[l]) for l in train_labels_full]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# MODEL BUILDER: EfficientNet-B4

def build_model(loss_type='focal'):
    """Build EfficientNet-B4 model. Returns (model, arcface_head_or_None)."""
    import timm
    backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)  # no classifier
    in_features = backbone.num_features  # 1792 for B4

    # Freeze first 4 blocks (blocks 0-3 in timm)
    # timm EfficientNet: conv_stem, bn1, blocks (list of block stages), conv_head, bn2
    frozen = 0
    for name, param in backbone.named_parameters():
        if name.startswith('blocks.') and int(name.split('.')[1]) < 4:
            param.requires_grad = False
            frozen += 1
        elif name.startswith('conv_stem') or name.startswith('bn1'):
            param.requires_grad = False
            frozen += 1

    arcface_head = None

    if loss_type == 'arcface':
        # Backbone -> embedding (512d) -> ArcFace head
        classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        arcface_head = ArcFaceHead(512, num_classes, scale=30.0, margin=0.5)
        model = nn.Sequential(backbone, classifier)
    elif loss_type == 'supcon':
        # Backbone -> projection head (128d normalized) for contrastive
        # Then later we replace with classifier for CE fine-tuning
        classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
        )
        model = nn.Sequential(backbone, classifier)
    else:  # focal
        classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        model = nn.Sequential(backbone, classifier)

    model = model.to(DEVICE)
    if arcface_head is not None:
        arcface_head = arcface_head.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)")

    return model, arcface_head


# TRAINING FUNCTION (shared across loss variants)

def train_model(model, loss_type, arcface_head=None, supcon_stage1=False, num_epochs=NUM_EPOCHS):
    """Train a model with the given loss type. Returns best_val_f1 and history."""

    if loss_type == 'focal':
        criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    elif loss_type == 'arcface':
        criterion = ArcFaceLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    elif loss_type == 'supcon' and supcon_stage1:
        criterion = SupConLoss(temperature=0.07)
    else:  # supcon stage 2 or fallback
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    criterion_val = nn.CrossEntropyLoss()

    params = list(model.parameters())
    if arcface_head is not None:
        params += list(arcface_head.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0
    patience_counter = 0
    model_save_path = os.path.join(OUTPUT_DIR, f"best_model_{loss_type}.pth")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        if arcface_head is not None:
            arcface_head.train()
        train_loss, train_correct, train_total = 0, 0, 0

        if loss_type == 'supcon' and supcon_stage1:
            # Contrastive training with two views
            contrastive_dataset = EmbeddingDataset(train_data_full, train_labels_full, train_transform)
            contrastive_loader = DataLoader(contrastive_dataset, batch_size=BATCH_SIZE,
                                           sampler=sampler, num_workers=NUM_WORKERS)
            for view1, view2, lbl in contrastive_loader:
                view1, view2, lbl = view1.to(DEVICE), view2.to(DEVICE), lbl.to(DEVICE)
                optimizer.zero_grad()
                z1 = F.normalize(model(view1), dim=1)
                z2 = F.normalize(model(view2), dim=1)
                features = torch.cat([z1, z2], dim=0)
                labels_dup = torch.cat([lbl, lbl], dim=0)
                loss = criterion(features, labels_dup)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * view1.size(0)
                train_total += view1.size(0)
            train_loss /= max(train_total, 1)
            train_acc = 0  # No classification accuracy in stage 1
        else:
            for images, lbl in train_loader:
                images, lbl = images.to(DEVICE), lbl.to(DEVICE)

                r = random.random()
                if r < CUTMIX_PROB:
                    images, targets_a, targets_b, lam = apply_cutmix(images, lbl, CUTMIX_ALPHA)
                    optimizer.zero_grad()
                    if arcface_head is not None:
                        embeddings = model(images)
                        outputs = arcface_head(embeddings, targets_a)
                        outputs_b = arcface_head(embeddings, targets_b)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs_b, targets_b)
                    else:
                        outputs = model(images)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                elif r < CUTMIX_PROB + MIXUP_PROB:
                    images, targets_a, targets_b, lam = apply_mixup(images, lbl, MIXUP_ALPHA)
                    optimizer.zero_grad()
                    if arcface_head is not None:
                        embeddings = model(images)
                        outputs = arcface_head(embeddings, targets_a)
                        outputs_b = arcface_head(embeddings, targets_b)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs_b, targets_b)
                    else:
                        outputs = model(images)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    optimizer.zero_grad()
                    if arcface_head is not None:
                        embeddings = model(images)
                        outputs = arcface_head(embeddings, lbl)
                    else:
                        outputs = model(images)
                    loss = criterion(outputs, lbl)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                train_correct += preds.eq(lbl).sum().item()
                train_total += images.size(0)

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

        scheduler.step(epoch)

        # Validate
        model.eval()
        if arcface_head is not None:
            arcface_head.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds_all, val_labels_all = [], []

        if not (loss_type == 'supcon' and supcon_stage1):
            with torch.no_grad():
                for images, lbl in val_loader:
                    images, lbl = images.to(DEVICE), lbl.to(DEVICE)
                    if arcface_head is not None:
                        embeddings = model(images)
                        outputs = arcface_head(embeddings)  # No labels = inference mode
                    else:
                        outputs = model(images)
                    loss = criterion_val(outputs, lbl)
                    val_loss += loss.item() * images.size(0)
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(lbl).sum().item()
                    val_total += images.size(0)
                    val_preds_all.extend(preds.cpu().numpy())
                    val_labels_all.extend(lbl.cpu().numpy())

            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
            val_macro_f1 = f1_score(val_labels_all, val_preds_all, average='macro', zero_division=0)
        else:
            val_loss, val_acc, val_macro_f1 = 0, 0, 0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_macro_f1)

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        marker = ""

        if loss_type == 'supcon' and supcon_stage1:
            # For stage 1, save every epoch (no F1 to track)
            torch.save(model.state_dict(), model_save_path)
            print(f"  Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} | LR: {lr:.6f} | {elapsed:.0f}s")
        else:
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                save_dict = {'model': model.state_dict()}
                if arcface_head is not None:
                    save_dict['arcface_head'] = arcface_head.state_dict()
                torch.save(save_dict, model_save_path)
                patience_counter = 0
                marker = " * best"
            else:
                patience_counter += 1

            print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_macro_f1:.4f} | "
                  f"LR: {lr:.6f} | {elapsed:.0f}s{marker}")

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return best_val_f1, history



# EVALUATION FUNCTION (with TTA)

def evaluate_model(model, loss_type, arcface_head=None):
    """Evaluate model on test set with TTA. Returns metrics dict."""
    model.eval()
    if arcface_head is not None:
        arcface_head.eval()

    def predict(loader):
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, lbl in loader:
                images = images.to(DEVICE)
                if arcface_head is not None:
                    embeddings = model(images)
                    outputs = arcface_head(embeddings)
                else:
                    outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbl.numpy())
                all_probs.extend(probs)
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    # Standard evaluation
    preds_no_tta, labels_arr, probs_no_tta = predict(test_loader)

    # TTA
    print(f"  Running TTA ({len(tta_transforms)} views)...")
    all_probs_tta = np.zeros_like(probs_no_tta)
    for t_idx, tta_t in enumerate(tta_transforms):
        tta_dataset = MinifigDataset(test_data, test_labels, tta_t)
        tta_loader = DataLoader(tta_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        _, _, probs_this = predict(tta_loader)
        all_probs_tta += probs_this

    all_probs_tta /= len(tta_transforms)
    preds_tta = all_probs_tta.argmax(axis=1)

    def compute_metrics(preds, labels, probs):
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

    metrics_no_tta = compute_metrics(preds_no_tta, labels_arr, probs_no_tta)
    metrics_tta = compute_metrics(preds_tta, labels_arr, all_probs_tta)

    return metrics_no_tta, metrics_tta, preds_tta, labels_arr, all_probs_tta



# TRAIN ALL THREE VARIANTS

all_results = {}

# --- 1. FOCAL LOSS ---
print("\n" + "=" * 80)
print("TRAINING V3-FOCAL (EfficientNet-B4 + Focal Loss)")
print("=" * 80)
model_focal, _ = build_model('focal')
best_f1_focal, hist_focal = train_model(model_focal, 'focal')
print(f"  Best val macro F1: {best_f1_focal:.4f}")

# Load best and evaluate
checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model_focal.pth"), map_location=DEVICE, weights_only=True)
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    model_focal.load_state_dict(checkpoint['model'])
else:
    model_focal.load_state_dict(checkpoint)
metrics_focal_no_tta, metrics_focal_tta, preds_focal, labels_focal, probs_focal = evaluate_model(model_focal, 'focal')
all_results['V3-Focal'] = {'no_tta': metrics_focal_no_tta, 'tta': metrics_focal_tta,
                            'history': hist_focal, 'preds': preds_focal, 'labels': labels_focal, 'probs': probs_focal}

# Clean up GPU memory
del model_focal
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- 2. ARCFACE LOSS ---
print("\n" + "=" * 80)
print("TRAINING V3-ARCFACE (EfficientNet-B4 + ArcFace Loss)")
print("=" * 80)
model_arc, arcface_head = build_model('arcface')
best_f1_arc, hist_arc = train_model(model_arc, 'arcface', arcface_head=arcface_head)
print(f"  Best val macro F1: {best_f1_arc:.4f}")

# Load best and evaluate
checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model_arcface.pth"), map_location=DEVICE, weights_only=True)
model_arc.load_state_dict(checkpoint['model'])
arcface_head.load_state_dict(checkpoint['arcface_head'])
metrics_arc_no_tta, metrics_arc_tta, preds_arc, labels_arc, probs_arc = evaluate_model(model_arc, 'arcface', arcface_head)
all_results['V3-ArcFace'] = {'no_tta': metrics_arc_no_tta, 'tta': metrics_arc_tta,
                              'history': hist_arc, 'preds': preds_arc, 'labels': labels_arc, 'probs': probs_arc}

del model_arc, arcface_head
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- 3. SUPCON + CE ---
print("\n" + "=" * 80)
print("TRAINING V3-SUPCON (EfficientNet-B4 + SupCon pre-train + CE fine-tune)")
print("=" * 80)

# Stage 1: Contrastive pre-training (shorter)
print("  Stage 1: Contrastive pre-training...")
model_supcon, _ = build_model('supcon')
supcon_epochs = min(10, NUM_EPOCHS)
_, hist_supcon_s1 = train_model(model_supcon, 'supcon', supcon_stage1=True, num_epochs=supcon_epochs)

# Stage 2: Replace projection head with classifier, fine-tune with CE
print("  Stage 2: Fine-tuning classifier with CE...")
# Get backbone from model_supcon (first module is backbone)
backbone = model_supcon[0]
in_features = backbone.num_features

# Build new classifier head
classifier_head = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes)
).to(DEVICE)

model_supcon_s2 = nn.Sequential(backbone, classifier_head).to(DEVICE)
# Save path for stage 2
best_f1_supcon, hist_supcon_s2 = train_model(model_supcon_s2, 'supcon_s2', num_epochs=NUM_EPOCHS)
print(f"  Best val macro F1: {best_f1_supcon:.4f}")

# Rename the saved model
supcon_s2_path = os.path.join(OUTPUT_DIR, "best_model_supcon_s2.pth")
supcon_final_path = os.path.join(OUTPUT_DIR, "best_model_supcon.pth")
if os.path.exists(supcon_s2_path):
    shutil.move(supcon_s2_path, supcon_final_path)

checkpoint = torch.load(supcon_final_path, map_location=DEVICE, weights_only=True)
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    model_supcon_s2.load_state_dict(checkpoint['model'])
else:
    model_supcon_s2.load_state_dict(checkpoint)
metrics_supcon_no_tta, metrics_supcon_tta, preds_supcon, labels_supcon, probs_supcon = evaluate_model(model_supcon_s2, 'supcon_s2')
all_results['V3-SupCon'] = {'no_tta': metrics_supcon_no_tta, 'tta': metrics_supcon_tta,
                             'history': hist_supcon_s2, 'preds': preds_supcon, 'labels': labels_supcon, 'probs': probs_supcon}

del model_supcon, model_supcon_s2
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# COMPARISON TABLE: ALL MODELS

print("\n" + "=" * 80)
print("FULL COMPARISON: ALL MODELS")
print("=" * 80)

# Historical results
baseline = {'Accuracy': 0.2930, 'Top-3 Accuracy': 0.4821, 'Top-5 Accuracy': 0.5818,
            'Macro F1': 0.3224, 'Weighted F1': 0.2730}
optb = {'Accuracy': 0.5885, 'Top-3 Accuracy': 0.8534, 'Top-5 Accuracy': 0.9290,
        'Macro F1': 0.5730, 'Weighted F1': 0.5777}
imp = {'Accuracy': 0.7440, 'Top-3 Accuracy': 0.9213, 'Top-5 Accuracy': 0.9585,
       'Macro F1': 0.6929, 'Weighted F1': 0.7487}
v2 = {'Accuracy': 0.7493, 'Top-3 Accuracy': 0.9029, 'Top-5 Accuracy': 0.9413,
      'Macro F1': 0.7083, 'Weighted F1': 0.7493}

header = f"{'Metric':<25} {'Baseline':>10} {'Opt B':>10} {'B Improv':>10} {'V2':>10}"
for name in all_results:
    header += f" {name:>12}"
print(header)
print("-" * len(header))

for metric in ['Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Macro F1', 'Weighted F1']:
    row = f"{metric:<25} {baseline[metric]:>10.4f} {optb[metric]:>10.4f} {imp[metric]:>10.4f} {v2[metric]:>10.4f}"
    for name in all_results:
        val = all_results[name]['tta'].get(metric, 0)
        row += f" {val:>12.4f}"
    print(row)

num_classes_row = f"{'Num Classes':<25} {'122':>10} {'28':>10} {'28':>10} {'37':>10}"
for name in all_results:
    num_classes_row += f" {'37':>12}"
print(num_classes_row)


# PICK BEST V3 VARIANT

best_variant = max(all_results.keys(), key=lambda k: all_results[k]['tta']['Macro F1'])
best_metrics = all_results[best_variant]['tta']

print(f"\nBest V3 variant: {best_variant}")
print(f"  Accuracy:       {best_metrics['Accuracy']:.4f}")
print(f"  Macro F1:       {best_metrics['Macro F1']:.4f}")
print(f"  Weighted F1:    {best_metrics['Weighted F1']:.4f}")
print(f"  Top-3 Accuracy: {best_metrics['Top-3 Accuracy']:.4f}")
print(f"  Top-5 Accuracy: {best_metrics['Top-5 Accuracy']:.4f}")


# DETAILED REPORT FOR BEST VARIANT

print(f"\n{'='*80}")
print(f"DETAILED RESULTS FOR {best_variant} (with TTA)")
print(f"{'='*80}")

best_preds = all_results[best_variant]['preds']
best_labels = all_results[best_variant]['labels']
best_probs = all_results[best_variant]['probs']

present_labels = sorted(set(best_labels) | set(best_preds))
present_names = [idx2cat[i] for i in present_labels]

report = classification_report(best_labels, best_preds, labels=present_labels,
                                target_names=present_names, zero_division=0, output_dict=True)
report_str = classification_report(best_labels, best_preds, labels=present_labels,
                                    target_names=present_names, zero_division=0)
print("\n" + report_str)

# Save classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
    f.write(f"V3 Results — Best Variant: {best_variant} (with TTA)\n\n")
    for k, v in best_metrics.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write(f"\n{report_str}")

# Save full comparison
with open(os.path.join(OUTPUT_DIR, "comparison_results.txt"), 'w') as f:
    f.write("V3 Full Comparison\n")
    f.write("=" * 100 + "\n")
    header_line = f"{'Metric':<25} {'Baseline':>10} {'Model V1':>10} {'Model V2':>10} {'Model V3':>10}"
    for name in all_results:
        header_line += f" {name:>12}"
    f.write(header_line + "\n")
    f.write("-" * len(header_line) + "\n")
    for metric in ['Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Macro F1', 'Weighted F1']:
        row = f"{metric:<25} {baseline[metric]:>10.4f} {optb[metric]:>10.4f} {imp[metric]:>10.4f} {v2[metric]:>10.4f}"
        for name in all_results:
            val = all_results[name]['tta'].get(metric, 0)
            row += f" {val:>12.4f}"
        f.write(row + "\n")
    f.write(f"\nBest variant: {best_variant}\n")


# PLOTS


# 1. Training curves for each variant
for variant_name, result in all_results.items():
    hist = result['history']
    safe_name = variant_name.replace('-', '_').lower()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot(hist['train_loss'], label='Train'); ax1.plot(hist['val_loss'], label='Val')
    ax1.set_title(f'{variant_name} — Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(hist['train_acc'], label='Train'); ax2.plot(hist['val_acc'], label='Val')
    ax2.set_title(f'{variant_name} — Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    ax3.plot(hist['val_f1'], label='Val Macro F1', color='green')
    ax3.set_title(f'{variant_name} — Val Macro F1'); ax3.set_xlabel('Epoch'); ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"training_curves_{safe_name}.png"), dpi=150)
    plt.close()

# 2. Confusion matrix for best variant
cm = confusion_matrix(best_labels, best_preds, labels=present_labels)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
fig, ax = plt.subplots(figsize=(26, 22))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=present_names, yticklabels=present_names, ax=ax, vmin=0, vmax=1)
ax.set_title(f'Normalized Confusion Matrix — {best_variant} (with TTA)')
ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.xticks(rotation=45, ha='right', fontsize=7); plt.yticks(rotation=0, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150); plt.close()

# 3. Per-class F1 for best variant
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
ax.set_xlabel('F1 Score'); ax.set_title(f'Per-Class F1 — {best_variant} (with TTA)')
ax.invert_yaxis(); ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1.png"), dpi=150); plt.close()

# 4. V3 variant comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
variant_names = list(all_results.keys())
x = np.arange(len(variant_names))
width = 0.2
metrics_to_plot = ['Accuracy', 'Macro F1', 'Weighted F1']
for i, metric in enumerate(metrics_to_plot):
    vals = [all_results[v]['tta'][metric] for v in variant_names]
    ax.bar(x + i * width, vals, width, label=metric)
ax.set_xlabel('V3 Variant'); ax.set_ylabel('Score')
ax.set_title('V3 Loss Function Comparison')
ax.set_xticks(x + width); ax.set_xticklabels(variant_names)
ax.legend(); ax.set_ylim(0.5, 1.0)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "v3_variant_comparison.png"), dpi=150); plt.close()

# 5. Confused pairs
print(f"\n{'='*80}")
print("TOP 15 MOST CONFUSED PAIRS")
print(f"{'='*80}")
full_cm = confusion_matrix(best_labels, best_preds, labels=range(num_classes))
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

# 6. Zero F1
zero_f1 = [(n, s) for n, f, s in per_class_f1 if f == 0]
print(f"\nCategories with F1=0: {len(zero_f1)}")
for n, s in zero_f1: print(f"  {n} (test: {s})")

# 7. Size buckets
print(f"\n{'='*80}")
print("PERFORMANCE BY CATEGORY SIZE")
print(f"{'='*80}")
test_counts = Counter(best_labels)
buckets = {'<10': [], '10-49': [], '50-99': [], '100-499': [], '500+': []}
for cat in present_names:
    idx = cat2idx[cat]
    support = test_counts.get(idx, 0)
    f1_val = report[cat]['f1-score']
    if support < 10: buckets['<10'].append(f1_val)
    elif support < 50: buckets['10-49'].append(f1_val)
    elif support < 100: buckets['50-99'].append(f1_val)
    elif support < 500: buckets['100-499'].append(f1_val)
    else: buckets['500+'].append(f1_val)

print(f"{'Bucket':<15} {'#Classes':>10} {'Avg F1':>10} {'Min F1':>10} {'Max F1':>10}")
print("-" * 57)
for bucket, f1s_list in buckets.items():
    if f1s_list:
        print(f"{bucket:<15} {len(f1s_list):>10} {np.mean(f1s_list):>10.4f} {min(f1s_list):>10.4f} {max(f1s_list):>10.4f}")

print(f"\nAll results saved to: {OUTPUT_DIR}")
print("Done!")
