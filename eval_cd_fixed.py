"""
eval_cd_fixed_v6.py — 配套 models/dd_resnet12_v6.py 的跨域评估脚本

支持5个目标域: EuroSAT / CUB / Places / Cars / ISIC
  - EuroSAT/ISIC: 通过 cd_dataset_adapter (原有路径不变)
  - CUB/Places/Cars: ImageFolder格式，自带CDCategoriesSampler

【重要】--gate-mode 和 --metric 必须和训练时完全一致。
smooth_aug 相关参数只影响训练，eval时不需要传入。

用法:
  python eval_cd_fixed_v6.py \
      --model-path ./save/v6_combo_1s/best.pth \
      --gate-mode hard --metric cosine \
      --eurosat-path data/EuroSAT \
      --isic-csv data/ISIC2018/.../GroundTruth.csv \
      --isic-img-dir data/ISIC2018/ISIC2018_Task3_Training_Input/ \
      --cub-path data/CUB_200_2011 \
      --places-path ~/autodl-tmp/CrossDomainFewShot/filelists/places/source/places365_standard/train \
      --cars-path ~/autodl-tmp/SVasP-main/datasets/cars/car_data/car_data/train \
      --both-shots --n-episodes 600 --ft-steps 0 --gpu 0
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dd_resnet12_v6 import DDResNet12, CosineProto, EuclideanProto


def compute_ci(data):
    a = np.array(data, dtype=float)
    return a.mean(), 1.96 * a.std() / np.sqrt(len(a))


# ─────────────────────────────────────────
# 数据集: CUB / ImageFolder通用(Places/Cars)
# ─────────────────────────────────────────
class CUBFlatDataset(Dataset):
    """CUB-200-2011: data_root/images/<class>/xxx.jpg"""
    def __init__(self, size=84, data_root='data/CUB_200_2011'):
        from torchvision.datasets import ImageFolder
        img_dir = os.path.join(data_root, 'images')
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"CUB images not found: {img_dir}")
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        folder = ImageFolder(img_dir)
        self.data  = [p for p, _ in folder.imgs]
        self.label = [l for _, l in folder.imgs]
        print(f"[CUB] {len(self.data)} images, "
              f"{len(set(self.label))} classes loaded.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        return self.transform(img), self.label[idx]


class ImageFolderFlatDataset(Dataset):
    """通用 ImageFolder 格式: data_root/<class_name>/xxx.jpg"""
    def __init__(self, size=84, data_root='', tag='Dataset'):
        from torchvision.datasets import ImageFolder
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"{tag} not found: {data_root}")
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        folder = ImageFolder(data_root)
        self.data  = [p for p, _ in folder.imgs]
        self.label = [l for _, l in folder.imgs]
        print(f"[{tag}] {len(self.data)} images, "
              f"{len(set(self.label))} classes loaded.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        return self.transform(img), self.label[idx]


# ─────────────────────────────────────────
# Episodic Sampler (用于CUB/Places/Cars)
# ─────────────────────────────────────────
class CDCategoriesSampler:
    def __init__(self, label, n_episodes, n_way, n_per):
        self.n_episodes = n_episodes
        self.n_way      = n_way
        self.n_per      = n_per
        label = np.array(label)
        self.m_ind = []
        for c in np.unique(label):
            ind = np.where(label == c)[0]
            if len(ind) >= n_per:
                self.m_ind.append(torch.from_numpy(ind))
        print(f"  有效类别数: {len(self.m_ind)} "
              f"(需要每类至少{n_per}张图)")

    def __len__(self): return self.n_episodes
    def __iter__(self):
        for _ in range(self.n_episodes):
            batch   = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                idx = self.m_ind[c]
                pos = torch.randperm(len(idx))[:self.n_per]
                batch.append(idx[pos])
            yield torch.stack(batch).reshape(-1)


# ─────────────────────────────────────────
# 统一数据加载入口
# ─────────────────────────────────────────
def get_loader(name, args, shot, way, n_query, n_episodes):
    n_per = shot + n_query
    size  = 84

    if name == 'eurosat':
        from datasets.cd_dataset_adapter import get_cd_loader as _get
        return _get('eurosat', shot=shot, n_way=way, n_query=n_query,
                    n_episodes=n_episodes, size=size,
                    num_workers=args.workers,
                    data_root=args.eurosat_path)

    elif name == 'isic':
        from datasets.cd_dataset_adapter import get_cd_loader as _get
        return _get('isic', shot=shot, n_way=way, n_query=n_query,
                    n_episodes=n_episodes, size=size,
                    num_workers=args.workers,
                    csv_path=args.isic_csv,
                    img_dir=args.isic_img_dir)

    elif name == 'cub':
        ds      = CUBFlatDataset(size=size, data_root=args.cub_path)
        sampler = CDCategoriesSampler(ds.label, n_episodes, way, n_per)
        return DataLoader(ds, batch_sampler=sampler,
                          num_workers=args.workers, pin_memory=True)

    elif name == 'places':
        ds      = ImageFolderFlatDataset(size=size,
                                          data_root=args.places_path,
                                          tag='Places')
        sampler = CDCategoriesSampler(ds.label, n_episodes, way, n_per)
        return DataLoader(ds, batch_sampler=sampler,
                          num_workers=args.workers, pin_memory=True)

    elif name == 'cars':
        ds      = ImageFolderFlatDataset(size=size,
                                          data_root=args.cars_path,
                                          tag='Cars')
        sampler = CDCategoriesSampler(ds.label, n_episodes, way, n_per)
        return DataLoader(ds, batch_sampler=sampler,
                          num_workers=args.workers, pin_memory=True)

    else:
        raise ValueError(f"Unknown dataset: {name}")


# ─────────────────────────────────────────
# 正确的切片 (shot-major, transpose修复)
# ─────────────────────────────────────────
def correct_split(data, shot, way, n_query):
    C, H, W = data.shape[1:]
    data    = data.view(way, shot + n_query, C, H, W)
    support = data[:, :shot].transpose(0, 1).reshape(shot * way, C, H, W)
    query   = data[:, shot:].reshape(way * n_query, C, H, W)
    label   = torch.arange(way).repeat_interleave(n_query)
    return support, query, label


# ─────────────────────────────────────────
# 模型加载 (gate_mode/metric 必须和训练时一致)
# ─────────────────────────────────────────
def load_model(model_path, gate_mode, metric, device):
    encoder = DDResNet12(drop_rate=0.0, dropblock_size=5,
                         gate_mode=gate_mode).to(device)

    if metric == 'cosine':
        head = CosineProto(init_scale=10.0).to(device)
    else:
        head = EuclideanProto().to(device)

    ckpt = torch.load(model_path, map_location=device,
                      weights_only=False)

    enc_ckpt  = {k.replace('encoder.', '', 1): v
                 for k, v in ckpt.items() if k.startswith('encoder.')}
    head_ckpt = {k.replace('cosine.',  '', 1): v
                 for k, v in ckpt.items() if k.startswith('cosine.')}

    me, ue = encoder.load_state_dict(enc_ckpt, strict=False)
    mh, uh = head.load_state_dict(head_ckpt,   strict=False)

    print(f"Loaded [{gate_mode} gate | {metric}]  "
          f"encoder: missing={len(me)} unexpected={len(ue)}  "
          f"head: missing={len(mh)} unexpected={len(uh)}")

    if gate_mode == 'soft':
        print(f"  gate_a = {torch.sigmoid(encoder.style_gate.alpha).item():.4f}")
    if metric == 'cosine':
        print(f"  cosine.scale = {head.scale.item():.4f}")

    encoder.eval()
    head.eval()
    return encoder, head


# ─────────────────────────────────────────
# 单episode评估 (含可选FT)
# ─────────────────────────────────────────
def eval_episode(encoder, head, support, query, query_label,
                 shot, way, metric, ft_steps=0, ft_lr=0.05):
    if ft_steps > 0 and metric == 'cosine':
        orig = head.scale.data.clone()
        head.scale.requires_grad_(True)
        opt = torch.optim.SGD([head.scale], lr=ft_lr)
        for _ in range(ft_steps):
            fs, _, _ = encoder(support)
            proto    = fs.reshape(shot, way, -1).mean(0)
            lbl      = torch.arange(way).repeat(shot).to(support.device)
            loss     = F.cross_entropy(head(fs, proto), lbl)
            opt.zero_grad(); loss.backward(); opt.step()
        head.scale.requires_grad_(False)
        head.scale.data.copy_(orig)

    with torch.no_grad():
        fs, _, _ = encoder(support)
        fq, _, _ = encoder(query)
        proto    = fs.reshape(shot, way, -1).mean(0)
        logits   = head(fq, proto)

    label = query_label.to(logits.device)
    preds = logits.argmax(1).cpu().numpy()
    labs  = label.cpu().numpy()
    acc   = (logits.argmax(1) == label).float().mean().item() * 100
    f1    = f1_score(labs, preds, average='macro') * 100
    return acc, f1


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder, head = load_model(args.model_path, args.gate_mode,
                               args.metric, device)

    # 根据传入的路径决定评估哪些数据集，保持固定顺序
    targets = []
    if args.eurosat_path: targets.append('eurosat')
    if args.cub_path:     targets.append('cub')
    if args.places_path:  targets.append('places')
    if args.cars_path:    targets.append('cars')
    if args.isic_csv:     targets.append('isic')

    if not targets:
        print("请至少指定一个数据集路径"); return

    shots = [1, 5] if args.both_shots else [args.shot]

    print(f"\n{'='*70}")
    print(f"v6跨域评估: {args.model_path}")
    print(f"数据集: {targets}  shots={shots}  ft_steps={args.ft_steps}")
    print(f"{'='*70}")

    results = {}
    for name in targets:
        for shot in shots:
            print(f"\n[{name}] {args.test_way}-way {shot}-shot "
                  f"({args.n_episodes} episodes) ...")
            try:
                loader = get_loader(name, args, shot, args.test_way,
                                    args.test_query, args.n_episodes)
            except Exception as e:
                print(f"  跳过: {e}"); continue

            acc_list, f1_list = [], []
            for i, (data, _) in enumerate(loader, 1):
                if i > args.n_episodes: break
                data = data.to(device)
                sup, qry, lbl = correct_split(
                    data, shot, args.test_way, args.test_query)
                acc, f1 = eval_episode(
                    encoder, head, sup, qry, lbl,
                    shot, args.test_way, args.metric,
                    ft_steps=args.ft_steps, ft_lr=args.ft_lr)
                acc_list.append(acc)
                f1_list.append(f1)

            acc_m, acc_ci = compute_ci(acc_list)
            f1_m, _       = compute_ci(f1_list)
            print(f"  acc={acc_m:.2f} ± {acc_ci:.2f}%  F1={f1_m:.2f}")
            results[(name, shot)] = (acc_m, acc_ci)

    # 机器可读输出行 (供自动化脚本抓取)
    if results:
        row = []
        for name in targets:
            for shot in shots:
                if (name, shot) in results:
                    m, ci = results[(name, shot)]
                    row.append(f"{m:.2f},{ci:.2f}")
                else:
                    row.append("NA,NA")
        print(f"\nCSV,{','.join(row)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',   required=True)
    p.add_argument('--gate-mode',    default='soft',
                   choices=['soft', 'hard', 'identity'],
                   help='必须和训练时--gate-mode一致')
    p.add_argument('--metric',       default='cosine',
                   choices=['cosine', 'euclidean'],
                   help='必须和训练时--metric一致')

    p.add_argument('--test-way',     type=int, default=5)
    p.add_argument('--test-query',   type=int, default=15)
    p.add_argument('--shot',         type=int, default=1)
    p.add_argument('--both-shots',   action='store_true',
                   help='同时评估1-shot和5-shot')
    p.add_argument('--n-episodes',   type=int, default=600)
    p.add_argument('--workers',      type=int, default=0)
    p.add_argument('--gpu',          default='0')
    p.add_argument('--ft-steps',     type=int, default=0)
    p.add_argument('--ft-lr',        type=float, default=0.05)

    # 数据集路径 (不传则跳过该数据集)
    p.add_argument('--eurosat-path', default='',
                   help='EuroSAT根目录')
    p.add_argument('--cub-path',     default='',
                   help='CUB_200_2011根目录 (含images/子目录)')
    p.add_argument('--places-path',  default='',
                   help='Places根目录 (ImageFolder格式)')
    p.add_argument('--cars-path',    default='',
                   help='Stanford Cars根目录 (ImageFolder格式)')
    p.add_argument('--isic-csv',     default='')
    p.add_argument('--isic-img-dir', default='')

    args = p.parse_args()
    main(args)
