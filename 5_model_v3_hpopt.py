"""
Model V3 Hyperparameter Tuning + 3-Fold Cross-Validation

Workflow:
1. OPTUNA: Search for best hyperparameters per loss variant (10 trials each)
2. CV: 3-fold cross-validation with best hyperparameters
3. REPORT: Final metrics + comparison vs original V3
"""
import json, os, time, warnings, random, math, shutil, copy
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, top_k_accuracy_score
)
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    print("Optuna is available")
except ImportError:
    print("ERROR: Optuna not installed. Install with: pip install optuna")
    exit(1)


# CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "model_v3_hpopt_results")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                       else "cpu")
print(f"Using device: {DEVICE}")

# Fixed hyperparameters
IMG_SIZE = 380
NUM_WORKERS = 2
MIN_SAMPLES_TARGET = 50

# HP Search config
HP_SEARCH_EPOCHS = 10
HP_SEARCH_TRIALS = 10
CV_EPOCHS = 20
CV_FOLDS = 3


# MERGE MAP & TOWN SPLIT (same as v3)

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

# LOSS FUNCTIONS (same as v3)

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
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale

class ArcFaceLoss(nn.Module):
    def __init__(self, weight=None, label_smoothing=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        return self.ce(logits, targets)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.clamp(mask_pos_pairs, min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = -mean_log_prob_pos.mean()
        return loss


# AUGMENTATION

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

class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        padded = ImageOps.pad(img, (max_side, max_side), color=(255, 255, 255))
        return padded


# DATASETS

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


# LOAD DATA (ONCE, GLOBALLY)

print("="*80)
print("LOADING DATA")
print("="*80)

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

# Build labels
labels = [cat2idx[d['merged_category']] for d in valid_data]
label_counts = Counter(labels)
small_idx = [i for i, l in enumerate(labels) if label_counts[l] < 7]
big_idx = [i for i, l in enumerate(labels) if label_counts[l] >= 7]
big_data = [valid_data[i] for i in big_idx]
big_labels = [labels[i] for i in big_idx]

# Stratified split: 70% train, 15% val, 15% test (on big data)

train_data_big, temp_data, train_labels_big, temp_labels = train_test_split(
    big_data, big_labels, test_size=0.3, random_state=42, stratify=big_labels)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
train_data = train_data_big + [valid_data[i] for i in small_idx]
train_labels = train_labels_big + [labels[i] for i in small_idx]
train_val_data = train_data + val_data
train_val_labels = train_labels + val_labels
print(f"Train+Val: {len(train_val_data)}, Test: {len(test_data)}")


# TRANSFORM DEFINITIONS (will parameterize augmentation later)

def get_train_transform(mixup_prob=0.3, cutmix_prob=0.3):
    return transforms.Compose([
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

print("\nData loaded. Ready for Phase 1: HP Search")
print("="*80)


# MODEL BUILDING (parameterized by hyperparameters)

def build_model_hp(loss_type, hp, num_classes):
    """Build EfficientNet-B4 with HP-specific architecture.
    
    hp dict keys:
    - dropout1: first dropout after linear
    - dropout2: second dropout
    - arc_scale, arc_margin: ArcFace specific
    """
    import os
    HF_TOKEN = os.environ.get('HF_TOKEN')
    if HF_TOKEN:
        print("HF_TOKEN detected for Hugging Face download")
    else:
        print("HF_TOKEN not set. Hugging Face may rate-limit downloads")
        
    import timm
    backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
    in_features = backbone.num_features  # 1792
    
    frozen = 0
    for name, param in backbone.named_parameters():
        if name.startswith('blocks.') and int(name.split('.')[1]) < 4:
            param.requires_grad = False
            frozen += 1
        elif name.startswith('conv_stem') or name.startswith('bn1'):
            param.requires_grad = False
            frozen += 1
    
    dropout1 = hp.get('dropout1', 0.4)
    dropout2 = hp.get('dropout2', 0.2)
    arcface_head = None
    
    if loss_type == 'arcface':
        classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout2),
        )
        arc_scale = hp.get('arc_scale', 30.0)
        arc_margin = hp.get('arc_margin', 0.5)
        arcface_head = ArcFaceHead(512, num_classes, scale=arc_scale, margin=arc_margin)
        model = nn.Sequential(backbone, classifier)
    elif loss_type == 'supcon':
        classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(512, 128),
        )
        model = nn.Sequential(backbone, classifier)
    else:  # focal
        classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(512, num_classes)
        )
        model = nn.Sequential(backbone, classifier)
    
    model = model.to(DEVICE)
    if arcface_head is not None:
        arcface_head = arcface_head.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model, arcface_head

# TRAINING FUNCTION (with Optuna pruning support)

def train_model_hp(model, loss_type, train_loader, val_loader,
                   train_data_full, train_labels_full, hp,
                   arcface_head=None, supcon_stage1=False,
                   num_epochs=20, trial=None, verbose=True):
    """Train model with HP.
    
    trial: Optuna Trial object (optional, for pruning)
    Returns: (best_val_f1, history)
    """
    
    # Loss function with HP
    class_weights = torch.zeros(num_classes)
    train_counts = Counter(train_labels_full)
    for i in range(num_classes):
        count = train_counts.get(i, 1)
        class_weights[i] = 1.0 / math.sqrt(count)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(DEVICE)
    
    if loss_type == 'focal':
        gamma = hp.get('focal_gamma', 2.0)
        label_smoothing = hp.get('label_smoothing', 0.1)
        criterion = FocalLoss(weight=class_weights, gamma=gamma, label_smoothing=label_smoothing)
    elif loss_type == 'arcface':
        label_smoothing = hp.get('label_smoothing', 0.1)
        criterion = ArcFaceLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif loss_type == 'supcon' and supcon_stage1:
        temp = hp.get('supcon_temp', 0.07)
        criterion = SupConLoss(temperature=temp)
    else:
        label_smoothing = hp.get('label_smoothing', 0.1)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    criterion_val = nn.CrossEntropyLoss()
    
    params = list(model.parameters())
    if arcface_head is not None:
        params += list(arcface_head.parameters())
    
    lr = hp.get('lr', 3e-4)
    weight_decay = hp.get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    
    t0_val = hp.get('t0', 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0_val, T_mult=2)
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0
    patience_counter = 0
    PATIENCE = 5
    
    cutmix_prob = hp.get('cutmix_prob', 0.3)
    mixup_prob = hp.get('mixup_prob', 0.3)
    cutmix_alpha = hp.get('cutmix_alpha', 1.0)
    mixup_alpha = hp.get('mixup_alpha', 0.2)
    
    for epoch in range(num_epochs):
        model.train()
        if arcface_head is not None:
            arcface_head.train()
        train_loss, train_total = 0, 0
        
        if loss_type == 'supcon' and supcon_stage1:
            # Contrastive training
            sample_weights = [1.0 / math.sqrt(train_counts[l]) for l in train_labels_full]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            contrastive_dataset = EmbeddingDataset(train_data_full, train_labels_full, get_train_transform())
            contrastive_loader = DataLoader(contrastive_dataset, batch_size=16, sampler=sampler, num_workers=NUM_WORKERS)
            
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
            val_loss, val_f1 = 0, 0
        else:
            for images, lbl in train_loader:
                images, lbl = images.to(DEVICE), lbl.to(DEVICE)
                
                r = random.random()
                if r < cutmix_prob:
                    images, targets_a, targets_b, lam = apply_cutmix(images, lbl, cutmix_alpha)
                    optimizer.zero_grad()
                    if arcface_head is not None:
                        embeddings = model(images)
                        outputs = arcface_head(embeddings, targets_a)
                        outputs_b = arcface_head(embeddings, targets_b)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs_b, targets_b)
                    else:
                        outputs = model(images)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                elif r < cutmix_prob + mixup_prob:
                    images, targets_a, targets_b, lam = apply_mixup(images, lbl, mixup_alpha)
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
                train_total += images.size(0)
            
            train_loss /= max(train_total, 1)
            
            # Validation
            model.eval()
            if arcface_head is not None:
                arcface_head.eval()
            val_loss, val_preds, val_labels_arr = 0, [], []
            
            if not (loss_type == 'supcon' and supcon_stage1):
                with torch.no_grad():
                    for images, lbl in val_loader:
                        images, lbl = images.to(DEVICE), lbl.to(DEVICE)
                        if arcface_head is not None:
                            embeddings = model(images)
                            outputs = arcface_head(embeddings)
                        else:
                            outputs = model(images)
                        loss = criterion_val(outputs, lbl)
                        val_loss += loss.item() * images.size(0)
                        _, preds = outputs.max(1)
                        val_preds.extend(preds.cpu().numpy())
                        val_labels_arr.extend(lbl.cpu().numpy())
                
                val_loss /= max(len(val_labels_arr), 1)
                val_f1 = f1_score(val_labels_arr, val_preds, average='macro', zero_division=0)
            else:
                val_loss, val_f1 = 0, 0
        
        scheduler.step(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        # Early stopping & checkpointing (for non-contrastive stages)
        if not (loss_type == 'supcon' and supcon_stage1):
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1}")
                break
        
        # Optuna trial reporting
        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if verbose and (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
    
    return best_val_f1, history

print("\nModel & training functions ready.")
print("="*80)


# PHASE 1: OPTUNA HYPERPARAMETER SEARCH

print("\nPHASE 1: OPTUNA HYPERPARAMETER SEARCH (10 trials per variant)")
print("="*80)

# For HP search, use a fixed 70/30 split on classes large enough for a
# stratified train/validation split. Tiny classes stay in HP train only.
hp_counts = Counter(train_val_labels)
hp_small_idx = [i for i, l in enumerate(train_val_labels) if hp_counts[l] < 2]
hp_big_idx = [i for i, l in enumerate(train_val_labels) if hp_counts[l] >= 2]

hp_big_data = [train_val_data[i] for i in hp_big_idx]
hp_big_labels = [train_val_labels[i] for i in hp_big_idx]

hp_search_train_data, hp_search_val_data, hp_search_train_labels, hp_search_val_labels = train_test_split(
    hp_big_data, hp_big_labels, test_size=0.30, random_state=42, stratify=hp_big_labels)

hp_search_train_data += [train_val_data[i] for i in hp_small_idx]
hp_search_train_labels += [train_val_labels[i] for i in hp_small_idx]

print(f"HP Search: Train {len(hp_search_train_data)}, Val {len(hp_search_val_data)}")

train_transform = get_train_transform()
hp_search_train_dataset = MinifigDataset(hp_search_train_data, hp_search_train_labels, train_transform)
hp_search_val_dataset = MinifigDataset(hp_search_val_data, hp_search_val_labels, eval_transform)

# Create Optuna studies
sampler_optuna = optuna.samplers.TPESampler(seed=42)
study_focal = optuna.create_study(
    direction='maximize',
    sampler=sampler_optuna,
    pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=2)
)
study_arcface = optuna.create_study(
    direction='maximize',
    sampler=sampler_optuna,
    pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=2)
)
study_supcon = optuna.create_study(
    direction='maximize',
    sampler=sampler_optuna,
    pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=2)
)

def objective_focal(trial):
    # Define search space
    lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    label_smoothing = trial.suggest_categorical('label_smoothing', [0.05, 0.10, 0.15])
    dropout1 = trial.suggest_categorical('dropout1', [0.3, 0.4, 0.5])
    dropout2 = trial.suggest_categorical('dropout2', [0.1, 0.2, 0.3])
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
    mixup_prob = trial.suggest_float('mixup_prob', 0.1, 0.4)
    cutmix_prob = trial.suggest_float('cutmix_prob', 0.1, 0.4)
    t0 = trial.suggest_int('t0', 3, 7)
    
    hp = {
        'lr': lr, 'weight_decay': weight_decay, 'label_smoothing': label_smoothing,
        'dropout1': dropout1, 'dropout2': dropout2, 'focal_gamma': focal_gamma,
        'mixup_prob': mixup_prob, 'cutmix_prob': cutmix_prob, 't0': t0,
        'cutmix_alpha': 1.0, 'mixup_alpha': 0.2
    }
    
    # Create batch loaders
    sample_weights = [1.0 / math.sqrt(Counter(hp_search_train_labels)[l]) for l in hp_search_train_labels]
    sampler_batch = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    hp_train_loader = DataLoader(hp_search_train_dataset, batch_size=16, sampler=sampler_batch, num_workers=NUM_WORKERS)
    hp_val_loader = DataLoader(hp_search_val_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    
    model, _ = build_model_hp('focal', hp, num_classes)
    try:
        val_f1, _ = train_model_hp(model, 'focal', hp_train_loader, hp_val_loader,
                                   hp_search_train_data, hp_search_train_labels, hp,
                                   num_epochs=HP_SEARCH_EPOCHS, trial=trial, verbose=False)
        return val_f1
    except Exception as e:
        print(f"      Trial failed: {e}")
        return 0.0
    finally:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def objective_arcface(trial):
    lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    label_smoothing = trial.suggest_categorical('label_smoothing', [0.05, 0.10, 0.15])
    dropout1 = trial.suggest_categorical('dropout1', [0.3, 0.4, 0.5])
    dropout2 = trial.suggest_categorical('dropout2', [0.1, 0.2, 0.3])
    arc_scale = trial.suggest_categorical('arc_scale', [20, 30, 40])
    arc_margin = trial.suggest_float('arc_margin', 0.3, 0.55)
    mixup_prob = trial.suggest_float('mixup_prob', 0.1, 0.4)
    cutmix_prob = trial.suggest_float('cutmix_prob', 0.1, 0.4)
    t0 = trial.suggest_int('t0', 3, 7)
    
    hp = {
        'lr': lr, 'weight_decay': weight_decay, 'label_smoothing': label_smoothing,
        'dropout1': dropout1, 'dropout2': dropout2, 'arc_scale': arc_scale, 'arc_margin': arc_margin,
        'mixup_prob': mixup_prob, 'cutmix_prob': cutmix_prob, 't0': t0,
        'cutmix_alpha': 1.0, 'mixup_alpha': 0.2
    }
    
    sample_weights = [1.0 / math.sqrt(Counter(hp_search_train_labels)[l]) for l in hp_search_train_labels]
    sampler_batch = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    hp_train_loader = DataLoader(hp_search_train_dataset, batch_size=16, sampler=sampler_batch, num_workers=NUM_WORKERS)
    hp_val_loader = DataLoader(hp_search_val_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    
    model, arcface_head = build_model_hp('arcface', hp, num_classes)
    try:
        val_f1, _ = train_model_hp(model, 'arcface', hp_train_loader, hp_val_loader,
                                   hp_search_train_data, hp_search_train_labels, hp,
                                   arcface_head=arcface_head, num_epochs=HP_SEARCH_EPOCHS,
                                   trial=trial, verbose=False)
        return val_f1
    except Exception as e:
        print(f"      Trial failed: {e}")
        return 0.0
    finally:
        del model, arcface_head
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def objective_supcon(trial):
    lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    label_smoothing = trial.suggest_categorical('label_smoothing', [0.05, 0.10, 0.15])
    dropout1 = trial.suggest_categorical('dropout1', [0.3, 0.4, 0.5])
    dropout2 = trial.suggest_categorical('dropout2', [0.1, 0.2, 0.3])
    supcon_temp = trial.suggest_float('supcon_temp', 0.05, 0.15)
    supcon_epochs = trial.suggest_int('supcon_epochs', 5, 10)
    mixup_prob = trial.suggest_float('mixup_prob', 0.1, 0.4)
    cutmix_prob = trial.suggest_float('cutmix_prob', 0.1, 0.4)
    t0 = trial.suggest_int('t0', 3, 7)
    
    hp = {
        'lr': lr, 'weight_decay': weight_decay, 'label_smoothing': label_smoothing,
        'dropout1': dropout1, 'dropout2': dropout2, 'supcon_temp': supcon_temp,
        'mixup_prob': mixup_prob, 'cutmix_prob': cutmix_prob, 't0': t0,
        'cutmix_alpha': 1.0, 'mixup_alpha': 0.2, 'supcon_epochs': supcon_epochs
    }
    
    sample_weights = [1.0 / math.sqrt(Counter(hp_search_train_labels)[l]) for l in hp_search_train_labels]
    sampler_batch = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    hp_train_loader = DataLoader(hp_search_train_dataset, batch_size=16, sampler=sampler_batch, num_workers=NUM_WORKERS)
    hp_val_loader = DataLoader(hp_search_val_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    
    model, _ = build_model_hp('supcon', hp, num_classes)
    try:
        # Stage 1: Contrastive
        _, _ = train_model_hp(model, 'supcon', hp_train_loader, hp_val_loader,
                             hp_search_train_data, hp_search_train_labels, hp,
                             supcon_stage1=True, num_epochs=supcon_epochs, verbose=False)
        
        # Stage 2: CE fine-tuning
        backbone = model[0]
        classifier_head = nn.Sequential(
            nn.Dropout(hp['dropout1']),
            nn.Linear(backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(hp['dropout2']),
            nn.Linear(512, num_classes)
        ).to(DEVICE)
        model_s2 = nn.Sequential(backbone, classifier_head).to(DEVICE)
        
        val_f1, _ = train_model_hp(model_s2, 'supcon', hp_train_loader, hp_val_loader,
                                  hp_search_train_data, hp_search_train_labels, hp,
                                  num_epochs=HP_SEARCH_EPOCHS, trial=trial, verbose=False)
        return val_f1
    except Exception as e:
        print(f"      Trial failed: {e}")
        return 0.0
    finally:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Run searches
print("\nFocal Loss HP Search (10 trials)...")
study_focal.optimize(objective_focal, n_trials=HP_SEARCH_TRIALS, show_progress_bar=True)
best_hp_focal = study_focal.best_trial.params
best_val_focal = study_focal.best_trial.value
print(f"  Best Focal: F1={best_val_focal:.4f}")
print(f"  Best HPs: {best_hp_focal}")

print("\nArcFace Loss HP Search (10 trials)...")
study_arcface.optimize(objective_arcface, n_trials=HP_SEARCH_TRIALS, show_progress_bar=True)
best_hp_arcface = study_arcface.best_trial.params
best_val_arcface = study_arcface.best_trial.value
print(f"  Best ArcFace: F1={best_val_arcface:.4f}")
print(f"  Best HPs: {best_hp_arcface}")

print("\nSupCon Loss HP Search (10 trials)...")
study_supcon.optimize(objective_supcon, n_trials=HP_SEARCH_TRIALS, show_progress_bar=True)
best_hp_supcon = study_supcon.best_trial.params
best_val_supcon = study_supcon.best_trial.value
print(f"  Best SupCon: F1={best_val_supcon:.4f}")
print(f"  Best HPs: {best_hp_supcon}")

# Save HP search results
with open(os.path.join(OUTPUT_DIR, 'hp_search_results.txt'), 'w') as f:
    f.write("="*80 + "\n")
    f.write("OPTUNA HP SEARCH RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Focal Loss Best Val F1: {best_val_focal:.4f}\n")
    f.write(f"Best HPs:\n{best_hp_focal}\n\n")
    f.write(f"ArcFace Loss Best Val F1: {best_val_arcface:.4f}\n")
    f.write(f"Best HPs:\n{best_hp_arcface}\n\n")
    f.write(f"SupCon Loss Best Val F1: {best_val_supcon:.4f}\n")
    f.write(f"Best HPs:\n{best_hp_supcon}\n")

print("\nPhase 1 complete. HP search results saved.")
print("="*80)


# PHASE 2: 3-FOLD CROSS-VALIDATION

print("\nPHASE 2: 3-FOLD CROSS-VALIDATION")
print("="*80)

# Use train+val data for CV, keep test fixed
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
cv_results = {
    'fold_metrics': [],  # List of per-fold metric dicts
    'fold_histories': [],  # List of training histories
}

fold_iter = list(skf.split(train_val_data, train_val_labels))

for fold_idx, (train_indices, val_indices) in enumerate(fold_iter):
    print(f"\nFold {fold_idx + 1}/{CV_FOLDS}")
    print("-" * 60)
    
    # Split data for this fold
    fold_train_data = [train_val_data[i] for i in train_indices]
    fold_train_labels = [train_val_labels[i] for i in train_indices]
    fold_val_data = [train_val_data[i] for i in val_indices]
    fold_val_labels = [train_val_labels[i] for i in val_indices]
    
    print(f"  Train: {len(fold_train_data)}, Val: {len(fold_val_data)}")
    
    # Create datasets
    fold_train_dataset = MinifigDataset(fold_train_data, fold_train_labels, train_transform)
    fold_val_dataset = MinifigDataset(fold_val_data, fold_val_labels, eval_transform)
    
    sample_weights_fold = [1.0 / math.sqrt(Counter(fold_train_labels)[l]) for l in fold_train_labels]
    sampler_fold = WeightedRandomSampler(sample_weights_fold, len(sample_weights_fold), replacement=True)
    fold_train_loader = DataLoader(fold_train_dataset, batch_size=16, sampler=sampler_fold, num_workers=NUM_WORKERS)
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    
    fold_metrics_dict = {}
    
    # Train Focal with best HP
    print(f"  Training Focal Loss...")
    model_focal, _ = build_model_hp('focal', best_hp_focal, num_classes)
    val_f1_focal, hist_focal = train_model_hp(model_focal, 'focal', fold_train_loader, fold_val_loader,
                                              fold_train_data, fold_train_labels, best_hp_focal,
                                              num_epochs=CV_EPOCHS, verbose=False)
    fold_metrics_dict['focal_f1'] = val_f1_focal
    torch.save(model_focal.state_dict(), os.path.join(OUTPUT_DIR, f'best_model_focal_fold{fold_idx}.pth'))
    del model_focal
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Train ArcFace with best HP
    print(f"  Training ArcFace Loss...")
    model_arc, arcface_head = build_model_hp('arcface', best_hp_arcface, num_classes)
    val_f1_arc, hist_arc = train_model_hp(model_arc, 'arcface', fold_train_loader, fold_val_loader,
                                          fold_train_data, fold_train_labels, best_hp_arcface,
                                          arcface_head=arcface_head, num_epochs=CV_EPOCHS, verbose=False)
    fold_metrics_dict['arcface_f1'] = val_f1_arc
    del model_arc, arcface_head
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Train SupCon with best HP
    print(f"  Training SupCon Loss...")
    model_supcon, _ = build_model_hp('supcon', best_hp_supcon, num_classes)
    supcon_epochs = best_hp_supcon.get('supcon_epochs', 10)
    _, _ = train_model_hp(model_supcon, 'supcon', fold_train_loader, fold_val_loader,
                         fold_train_data, fold_train_labels, best_hp_supcon,
                         supcon_stage1=True, num_epochs=supcon_epochs, verbose=False)
    
    backbone = model_supcon[0]
    classifier_head = nn.Sequential(
        nn.Dropout(best_hp_supcon['dropout1']),
        nn.Linear(backbone.num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(best_hp_supcon['dropout2']),
        nn.Linear(512, num_classes)
    ).to(DEVICE)
    model_supcon_s2 = nn.Sequential(backbone, classifier_head).to(DEVICE)
    val_f1_supcon, hist_supcon = train_model_hp(model_supcon_s2, 'supcon', fold_train_loader, fold_val_loader,
                                               fold_train_data, fold_train_labels, best_hp_supcon,
                                               num_epochs=CV_EPOCHS, verbose=False)
    fold_metrics_dict['supcon_f1'] = val_f1_supcon
    del model_supcon, model_supcon_s2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"  Fold Results: Focal F1={val_f1_focal:.4f}, ArcFace F1={val_f1_arc:.4f}, SupCon F1={val_f1_supcon:.4f}")
    cv_results['fold_metrics'].append(fold_metrics_dict)

# Report fold results
print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
print(f"{'Fold':<8} {'Focal F1':>12} {'ArcFace F1':>12} {'SupCon F1':>12}")
print("-" * 50)

for fold_idx, metrics in enumerate(cv_results['fold_metrics']):
    print(f"{fold_idx + 1:<8} {metrics['focal_f1']:>12.4f} {metrics['arcface_f1']:>12.4f} {metrics['supcon_f1']:>12.4f}")

focal_folds = [m['focal_f1'] for m in cv_results['fold_metrics']]
arcface_folds = [m['arcface_f1'] for m in cv_results['fold_metrics']]
supcon_folds = [m['supcon_f1'] for m in cv_results['fold_metrics']]

print("-" * 50)
print(f"{'Mean':<8} {np.mean(focal_folds):>12.4f} {np.mean(arcface_folds):>12.4f} {np.mean(supcon_folds):>12.4f}")
print(f"{'Std':<8} {np.std(focal_folds):>12.4f} {np.std(arcface_folds):>12.4f} {np.std(supcon_folds):>12.4f}")

# Save CV results
with open(os.path.join(OUTPUT_DIR, 'cv_results.txt'), 'w') as f:
    f.write("="*80 + "\n")
    f.write("3-FOLD CROSS-VALIDATION RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Fold':<8} {'Focal F1':>12} {'ArcFace F1':>12} {'SupCon F1':>12}\n")
    f.write("-" * 50 + "\n")
    for fold_idx, metrics in enumerate(cv_results['fold_metrics']):
        f.write(f"{fold_idx + 1:<8} {metrics['focal_f1']:>12.4f} {metrics['arcface_f1']:>12.4f} {metrics['supcon_f1']:>12.4f}\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Mean':<8} {np.mean(focal_folds):>12.4f} {np.mean(arcface_folds):>12.4f} {np.mean(supcon_folds):>12.4f}\n")
    f.write(f"{'Std':<8} {np.std(focal_folds):>12.4f} {np.std(arcface_folds):>12.4f} {np.std(supcon_folds):>12.4f}\n")

print("\nPhase 2 complete. CV results saved.")
print("="*80)



# PHASE 3: FINAL COMPARISON WITH ORIGINAL V3

print("\nPHASE 3: FINAL COMPARISON")
print("="*80)

# Original V3 results (hardcoded from optionB_v3_results/comparison_results.txt)
original_v3 = {
    'V3-Focal': {'Accuracy': 0.7735, 'Macro F1': 0.7315, 'Weighted F1': 0.7736},
    'V3-ArcFace': {'Accuracy': 0.7436, 'Macro F1': 0.6913, 'Weighted F1': 0.7345},
    'V3-SupCon': {'Accuracy': 0.7950, 'Macro F1': 0.7406, 'Weighted F1': 0.7951},
}

summary = {
    'Method': ['Baseline (Fixed HP)', 'HP-Tuned CV (Avg)', 'HP-Tuned CV (Best Fold)'],
    'Focal F1': [
        original_v3['V3-Focal']['Weighted F1'],
        np.mean(focal_folds),
        np.max(focal_folds)
    ],
    'ArcFace F1': [
        original_v3['V3-ArcFace']['Weighted F1'],
        np.mean(arcface_folds),
        np.max(arcface_folds)
    ],
    'SupCon F1': [
        original_v3['V3-SupCon']['Weighted F1'],
        np.mean(supcon_folds),
        np.max(supcon_folds)
    ],
}

print(f"{'Method':<30} {'Focal F1':>15} {'ArcFace F1':>15} {'SupCon F1':>15}")
print("-" * 80)
for i, method in enumerate(summary['Method']):
    print(f"{method:<30} {summary['Focal F1'][i]:>15.4f} {summary['ArcFace F1'][i]:>15.4f} {summary['SupCon F1'][i]:>15.4f}")

print("\nImprovement Analysis (HP-Tuned avg vs Baseline):")
print("-" * 80)
focal_improve = (summary['Focal F1'][1] - summary['Focal F1'][0]) / summary['Focal F1'][0] * 100
arcface_improve = (summary['ArcFace F1'][1] - summary['ArcFace F1'][0]) / summary['ArcFace F1'][0] * 100
supcon_improve = (summary['SupCon F1'][1] - summary['SupCon F1'][0]) / summary['SupCon F1'][0] * 100

print(f"Focal: {focal_improve:+.2f}%")
print(f"ArcFace: {arcface_improve:+.2f}%")
print(f"SupCon: {supcon_improve:+.2f}%")

# Save final comparison
with open(os.path.join(OUTPUT_DIR, 'final_comparison.txt'), 'w') as f:
    f.write("="*80 + "\n")
    f.write("FINAL COMPARISON: HP-TUNED vs MODEL V3\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Method':<30} {'Focal F1':>15} {'ArcFace F1':>15} {'SupCon F1':>15}\n")
    f.write("-" * 80 + "\n")
    for i, method in enumerate(summary['Method']):
        f.write(f"{method:<30} {summary['Focal F1'][i]:>15.4f} {summary['ArcFace F1'][i]:>15.4f} {summary['SupCon F1'][i]:>15.4f}\n")
    f.write("\n")
    f.write("Improvement vs Baseline (%):\n")
    f.write("-" * 80 + "\n")
    f.write(f"Focal:   {focal_improve:+.2f}%\n")
    f.write(f"ArcFace: {arcface_improve:+.2f}%\n")
    f.write(f"SupCon:  {supcon_improve:+.2f}%\n")
    f.write("\n" + "="*80 + "\n")
    f.write("BEST HYPERPARAMETERS FOUND\n")
    f.write("="*80 + "\n\n")
    f.write("Focal Loss:\n")
    for k, v in best_hp_focal.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nArcFace Loss:\n")
    for k, v in best_hp_arcface.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nSupCon Loss:\n")
    for k, v in best_hp_supcon.items():
        f.write(f"  {k}: {v}\n")

print("\nPhase 3 complete. All results saved.")
print("="*80)


# SUMMARY

print("\n" + "="*80)
print("HP TUNING + CV PIPELINE COMPLETE")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"\nOutput files:")
print(f"  - hp_search_results.txt")
print(f"  - cv_results.txt")
print(f"  - final_comparison.txt")
print(f"  - best_model_*.pth (fold checkpoints)")
print(f"\nKey Metrics:")
print(f"  Focal (CV avg):   {np.mean(focal_folds):.4f} ± {np.std(focal_folds):.4f} (vs baseline {summary['Focal F1'][0]:.4f})")
print(f"  ArcFace (CV avg): {np.mean(arcface_folds):.4f} ± {np.std(arcface_folds):.4f} (vs baseline {summary['ArcFace F1'][0]:.4f})")
print(f"  SupCon (CV avg):  {np.mean(supcon_folds):.4f} ± {np.std(supcon_folds):.4f} (vs baseline {summary['SupCon F1'][0]:.4f})")
print("\n" + "="*80)

