import os
import io
import zipfile
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
from torchvision import transforms


# Class name mapping (sanitize for filesystem safety)
CLASS_MAP = {
    "almond": "almond",
    "coconut": "coconut",
    "sunflower": "sunflower",
    "tomato": "tomato",
    "vigna-radiati(Mung)": "vigna_radiati_mung",
    "wheat": "wheat",
}

CLASS_NAMES = sorted(CLASS_MAP.values())
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Heavy augmentation pipeline for data enrichment (addresses limited data problem)."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CropZipDataset(Dataset):

    def __init__(
        self,
        zip_path: str,
        transform=None,
        max_per_class: Optional[int] = None,
        allowed_classes: Optional[List[str]] = None,
    ):
        self.zip_path = zip_path
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []  # (zip_member_name, class_idx)
        self.class_counts: Counter = Counter()

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()

        for name in members:
            low = name.lower()
            if not low.endswith((".jpg", ".jpeg", ".png")):
                continue
            if "Agricultural-crops/" not in name:
                continue
            parts = name.split("/")
            if len(parts) < 3:
                continue
            raw_cls = parts[1]
            if raw_cls not in CLASS_MAP:
                continue
            cls = CLASS_MAP[raw_cls]
            if allowed_classes and cls not in allowed_classes:
                continue
            if max_per_class is not None and self.class_counts[cls] >= max_per_class:
                continue
            self.samples.append((name, CLASS_TO_IDX[cls]))
            self.class_counts[cls] += 1

        print(f"[CropZipDataset] Loaded {len(self.samples)} images from {zip_path}")
        print(f"  Class distribution: {dict(self.class_counts)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        member, label = self.samples[idx]
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            data = zf.read(member)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class CropFolderDataset(Dataset):

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Counter = Counter()

        for cls_name in CLASS_NAMES:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                # try original names
                alt = self.root / cls_name.replace("_", "-")
                if not alt.is_dir():
                    continue
                cls_dir = alt
            for p in cls_dir.glob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((p, CLASS_TO_IDX[cls_name]))
                    self.class_counts[cls_name] += 1

        print(f"[CropFolderDataset] Loaded {len(self.samples)} images from {root}")
        print(f"  Class distribution: {dict(self.class_counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def create_dataloaders(
    zip_path: str = "/home/workdir/attachments/Crop.zip",
    data_dir: Optional[str] = None,
    batch_size: int = 16,
    img_size: int = 224,
    max_per_class: Optional[int] = 100,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data_dir and Path(data_dir).exists() and any(Path(data_dir).iterdir()):
        full_ds = CropFolderDataset(data_dir, transform=None)
    else:
        full_ds = CropZipDataset(zip_path, transform=None, max_per_class=max_per_class)

    # Stratified split indices
    labels = [s[1] for s in full_ds.samples]
    indices = list(range(len(full_ds)))
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=labels, random_state=seed
    )

    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)

    # Subset wrappers that apply transforms
    class TransformSubset(Dataset):
        def __init__(self, base, idxs, transform):
            self.base = base
            self.idxs = idxs
            self.transform = transform

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            # temporarily set transform
            orig = self.base.transform
            self.base.transform = self.transform
            item = self.base[self.idxs[i]]
            self.base.transform = orig
            return item

    train_ds = TransformSubset(full_ds, train_idx, train_transform)
    val_ds = TransformSubset(full_ds, val_idx, val_transform)

    # Class-balanced sampler for training (enrich minority classes)
    train_labels = [labels[i] for i in train_idx]
    class_sample_count = Counter(train_labels)
    weights = [1.0 / class_sample_count[lab] for lab in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    info = {
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_counts": dict(full_ds.class_counts),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    return train_loader, val_loader, info


def analyze_dataset(zip_path: str = "/home/workdir/attachments/Crop.zip") -> Dict:
    """Quick dataset statistics for the project report."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))
                   and "Agricultural-crops/" in m]
    counts = Counter()
    sizes = []
    for m in members:
        parts = m.split("/")
        if len(parts) >= 3 and parts[1] in CLASS_MAP:
            counts[CLASS_MAP[parts[1]]] += 1
    return {
        "total_images": sum(counts.values()),
        "num_classes": len(counts),
        "class_distribution": dict(counts),
        "imbalance_ratio": max(counts.values()) / max(1, min(counts.values())),
    }
