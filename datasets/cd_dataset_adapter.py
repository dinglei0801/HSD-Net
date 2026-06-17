"""
cd_dataset_adapter.py

把BSCD-FSL原始格式的EuroSAT/ISIC适配成
标准episodic DataLoader，兼容现有训练代码。

用法:
    from datasets.cd_dataset_adapter import get_cd_loader
    loader = get_cd_loader('eurosat', shot=1, n_way=5,
                           n_query=15, n_episodes=600, size=84)
    for data, _ in loader:
        data_shot, data_query = data[:shot*n_way], data[shot*n_way:]
"""

import os
import sys
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ─────────────────────────────────────────────
# 路径配置  ← 修改这里指向你的数据目录
# ─────────────────────────────────────────────
EUROSAT_PATH    = os.environ.get('EUROSAT_PATH',    'data/EuroSAT')
ISIC_PATH       = os.environ.get('ISIC_PATH',       'data/ISIC')
ISIC_CSV        = os.environ.get('ISIC_CSV',
    'data/ISIC/ISIC2018_Task3_Training_GroundTruth/'
    'ISIC2018_Task3_Training_GroundTruth.csv')
ISIC_IMG_DIR    = os.environ.get('ISIC_IMG_DIR',
    'data/ISIC/ISIC2018_Task3_Training_Input/')


# ─────────────────────────────────────────────
# 1.  EuroSAT  (10 classes, ImageFolder format)
# ─────────────────────────────────────────────
class EuroSATFlatDataset(Dataset):
    """
    Flat dataset: returns (tensor, int_label).
    Compatible with CategoriesSampler via self.label list.
    """
    def __init__(self, size=84, data_root=None):
        from torchvision.datasets import ImageFolder
        root = data_root or EUROSAT_PATH
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"EuroSAT not found at '{root}'.\n"
                f"Set env var EUROSAT_PATH or pass data_root=.")

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                  [0.229,0.224,0.225]),
        ])
        folder = ImageFolder(root)
        # folder.imgs: list of (path, class_idx)
        self.data  = [p for p, _ in folder.imgs]
        self.label = [l for _, l in folder.imgs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        return self.transform(img), self.label[idx]


# ─────────────────────────────────────────────
# 2.  ISIC  (7 classes, CSV + jpg)
# ─────────────────────────────────────────────
class ISICFlatDataset(Dataset):
    """
    Flat dataset: returns (tensor, int_label).
    Reads ISIC2018 Task3 CSV (one-hot labels).
    """
    def __init__(self, size=84, data_root=None,
                 csv_path=None, img_dir=None):
        import pandas as pd

        csv_path = csv_path or ISIC_CSV
        img_dir  = img_dir  or ISIC_IMG_DIR

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"ISIC CSV not found: '{csv_path}'.\n"
                f"Set env var ISIC_CSV or pass csv_path=.")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"ISIC image dir not found: '{img_dir}'.\n"
                f"Set env var ISIC_IMG_DIR or pass img_dir=.")

        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                  [0.229,0.224,0.225]),
        ])

        # Read CSV  (skip first row = column headers already in row 0)
        df = pd.read_csv(csv_path)
        # columns: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
        label_cols = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
        # if columns don't exist, fall back to positional
        if not all(c in df.columns for c in label_cols):
            df = pd.read_csv(csv_path, skiprows=[0], header=None)
            img_names = df.iloc[:, 0].values
            onehot    = df.iloc[:, 1:].values.astype(float)
        else:
            img_names = df['image'].values
            onehot    = df[label_cols].values.astype(float)

        labels = (onehot != 0).argmax(axis=1)

        self.data  = []
        self.label = []
        for name, lbl in zip(img_names, labels):
            for ext in ('.jpg', '.jpeg', '.JPG', '.JPEG'):
                p = os.path.join(img_dir, str(name) + ext)
                if os.path.exists(p):
                    self.data.append(p)
                    self.label.append(int(lbl))
                    break

        if len(self.data) == 0:
            raise RuntimeError(
                f"No ISIC images loaded from '{img_dir}'. "
                "Check paths and file extensions.")
        print(f"[ISIC] {len(self.data)} images, "
              f"{len(set(self.label))} classes loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        return self.transform(img), self.label[idx]


# ─────────────────────────────────────────────
# 3.  CategoriesSampler  (same as your existing one)
#     Inlined here so cd_dataset_adapter has no
#     dependency on your datasets/samplers.py
# ─────────────────────────────────────────────
class CDCategoriesSampler:
    """
    Episodic sampler: each iteration yields indices for
    n_way classes × (shot + n_query) images.
    """
    def __init__(self, label, n_episodes, n_way, n_per):
        self.n_episodes = n_episodes
        self.n_way      = n_way
        self.n_per      = n_per

        label = np.array(label)
        self.m_ind = []
        for c in np.unique(label):
            ind = np.argwhere(label == c).reshape(-1)
            self.m_ind.append(torch.from_numpy(ind))

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                idx = self.m_ind[c]
                pos = torch.randperm(len(idx))[:self.n_per]
                batch.append(idx[pos])
            yield torch.stack(batch).reshape(-1)


# ─────────────────────────────────────────────
# 4.  Public API
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 3.  CUB-200-2011
# ─────────────────────────────────────────────
class CUBFlatDataset(Dataset):
    """
    CUB-200-2011 loader for few-shot cross-domain evaluation.
    Expects: data_root/images/001.Black_footed_Albatross/...
    """
    def __init__(self, size=84, data_root=None):
        from torchvision.datasets import ImageFolder
        root = data_root or os.environ.get('CUB_PATH', 'data/CUB_200_2011')
        img_dir = os.path.join(root, 'images')
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"CUB images not found at {img_dir}\n"
                f"Set env var CUB_PATH or pass data_root=")

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        folder = ImageFolder(img_dir)
        self.data  = [p for p, _ in folder.imgs]
        self.label = [l for _, l in folder.imgs]
        print(f"[CUB] {len(self.data)} images, {len(set(self.label))} classes loaded.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        return self.transform(img), self.label[idx]

def get_cd_loader(name, shot=1, n_way=5, n_query=15,
                  n_episodes=600, size=84,
                  num_workers=4, **kwargs):
    """
    Returns an episodic DataLoader for cross-domain evaluation.

    Args:
        name       : 'eurosat' | 'isic'
        shot       : support shots per class
        n_way      : number of classes per episode
        n_query    : query samples per class
        n_episodes : total test episodes (600 per reviewer requirement)
        size       : image resize (84 for mini/tiered, 32 for cifarfs)
        **kwargs   : passed to dataset constructor (data_root, csv_path…)

    Returns:
        DataLoader where each batch is [n_way*(shot+n_query), C, H, W]
        Support: batch[:n_way*shot], Query: batch[n_way*shot:]
    """
    name = name.lower()
    if name == 'eurosat':
        ds = EuroSATFlatDataset(size=size,
                                data_root=kwargs.get('data_root'))
    elif name == 'cub':
        ds = CUBFlatDataset(size=size,
                            data_root=kwargs.get('data_root'))
    elif name == 'isic':
        ds = ISICFlatDataset(size=size,
                             data_root=kwargs.get('data_root'),
                             csv_path=kwargs.get('csv_path'),
                             img_dir=kwargs.get('img_dir'))
    else:
        raise ValueError(f"Unknown CD dataset: {name}. "
                         f"Choices: ['eurosat', 'isic']")

    n_per    = shot + n_query
    sampler  = CDCategoriesSampler(ds.label, n_episodes, n_way, n_per)
    loader   = DataLoader(
        dataset        = ds,
        batch_sampler  = sampler,
        num_workers    = num_workers,
        pin_memory     = True,
    )
    print(f"[CD-Loader] {name} | {n_episodes} episodes | "
          f"{n_way}-way {shot}-shot {n_query}-query")
    return loader
