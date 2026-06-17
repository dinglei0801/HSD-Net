"""
train_dd_eirm_v6.py — v5基础上新增smooth_style平滑增强 (针对EuroSAT优化)

相对v5的改动:
  - 模型: models/dd_resnet12_v6.py，新增StyleAugmentation.smooth_style()
  - 新增CLI: --smooth-aug/--no-smooth-aug, --smooth-kernel-size,
    --smooth-alpha-min/max, --smooth-prob
  - v5的所有消融开关(gate_mode/metric/ssl_coef/con_coef/dis_weight/
    style_aug)完全保留

v5h_combo + smooth_aug 运行示例(针对EuroSAT优化的推荐配置):
  python train_dd_eirm_v6.py --dataset mini --shot 1 \
      --irm-coef 0 --con-coef 0 --gate-mode hard \
      --smooth-aug --smooth-prob 0.5 \
      --save-path ./save/v6_combo_1s \
      --max-epoch 200 --gpu 0

如果想单独验证smooth_aug本身的贡献(不叠加v5h_combo):
  python train_dd_eirm_v6.py --dataset mini --shot 1 --irm-coef 0 \
      --smooth-aug --smooth-prob 0.5 \
      --save-path ./save/v6_smooth_only_1s \
      --max-epoch 200 --gpu 0
"""

import argparse
import datetime
import os
import os.path as osp
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, recall_score

from models.dd_resnet12_v6 import DDResNetEIRM


# ─────────────────────────────────────────
# Utils
# ─────────────────────────────────────────
def seed_torch(seed=1337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(path):
    os.makedirs(path, exist_ok=True)

class Averager:
    def __init__(self): self.n = 0; self.v = 0.
    def add(self, x):   self.v = (self.v*self.n + x)/(self.n+1); self.n += 1
    def item(self):     return self.v

def compute_ci(data):
    a = np.array(data, dtype=float)
    return a.mean(), 1.96*a.std()/np.sqrt(len(a))

class Timer:
    def __init__(self): self.o = datetime.datetime.now()
    def elapsed(self):
        s = int((datetime.datetime.now()-self.o).total_seconds())
        return f"{s//3600}h{(s%3600)//60}m" if s>=3600 else (f"{s//60}m" if s>=60 else f"{s}s")


# ─────────────────────────────────────────
# 【关键修复】cd_dataset_adapter正确切片
# ─────────────────────────────────────────
def correct_cd_split(data, shot, way, n_query):
    """
    cd_dataset_adapter数据布局: [class0的(shot+n_query)张][class1的...]...

    返回:
      support: (way*shot, C,H,W)
               顺序 = [shot-major, way-minor]，即
               [shot0: c0,c1,...,c(way-1)][shot1: c0,...,c(way-1)]...
               与 proto_logits 里 feat.reshape(shot,way,-1).mean(0)
               (标准CategoriesSampler约定) 完全一致
      query:   (way*n_query, C,H,W)
      query_label: arange(way).repeat_interleave(n_query)
    """
    C, H, W = data.shape[1:]
    data = data.view(way, shot + n_query, C, H, W)

    # support: (way, shot, C,H,W) -> transpose -> (shot, way, C,H,W) -> flatten
    support = data[:, :shot].transpose(0, 1).reshape(shot * way, C, H, W)

    query   = data[:, shot:].reshape(way * n_query, C, H, W)
    query_label = torch.arange(way).repeat_interleave(n_query)
    return support, query, query_label


# ─────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────
def get_loader(args, split):
    size = args.size
    if args.dataset == 'mini':
        from datasets.mini_imagenet import MiniImageNet
        ds = MiniImageNet(split, size)
    elif args.dataset == 'tiered':
        from datasets.tiered_imagenet import TieredImageNet
        ds = TieredImageNet(split, size)
    elif args.dataset == 'cifarfs':
        from datasets.cifarfs import CIFAR_FS
        ds = CIFAR_FS(split, size)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    from datasets.samplers import CategoriesSampler
    n_ep  = args.train_batch if split == 'train' else args.test_batch
    way   = args.train_way   if split == 'train' else args.test_way
    n_per = args.shot + (args.train_query if split == 'train' else args.test_query)
    sampler = CategoriesSampler(ds.label, n_ep, way, n_per)
    return DataLoader(ds, batch_sampler=sampler,
                      num_workers=args.worker, pin_memory=True)


def get_cd_loader(target_name, args):
    from datasets.cd_dataset_adapter import get_cd_loader as _get
    kwargs = {}
    if target_name == 'eurosat' and args.eurosat_path:
        kwargs['data_root'] = args.eurosat_path
    elif target_name == 'isic':
        if args.isic_csv:     kwargs['csv_path']  = args.isic_csv
        if args.isic_img_dir: kwargs['img_dir']   = args.isic_img_dir
    try:
        return _get(name=target_name, shot=args.shot,
                    n_way=args.test_way, n_query=args.test_query,
                    n_episodes=args.test_batch, size=args.size,
                    num_workers=args.worker, **kwargs)
    except Exception as e:
        print(f"[Warning] Skipping '{target_name}': {e}")
        return None


# ─────────────────────────────────────────
# Train
# ─────────────────────────────────────────
def train_epoch(args, model, loader, optimizer, scaler, epoch):
    model.train()
    tl = Averager(); ta = Averager()

    for batch in loader:
        data, _ = [x.cuda() for x in batch]
        p = args.shot * args.train_way
        shot, query = data[:p], data[p:]

        with autocast(device_type='cuda', dtype=torch.float16):
            total, L_cls, L_dis, L_eirm, acc, w_irm = model.forward_train(
                shot, query, args.shot, args.train_way, args.train_query, epoch
            )

        optimizer.zero_grad()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        tl.add(total.item())
        ta.add(acc)

    return tl.item(), ta.item()


# ─────────────────────────────────────────
# Validate (标准CategoriesSampler，用于训练时的val_loader)
# ─────────────────────────────────────────
def validate(args, model, loader, n_ep=600):
    model.eval()
    acc_list, f1_list, rec_list = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if i > n_ep: break
            data, _ = [x.cuda() for x in batch]
            p = args.shot * args.test_way
            shot, query = data[:p], data[p:]
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model.forward_eval(
                    shot, query, args.shot, args.test_way, args.test_query)
            label = torch.arange(args.test_way).repeat(args.test_query).cuda()
            preds = logits.argmax(1).cpu().numpy()
            labs  = label.cpu().numpy()
            acc_list.append((logits.argmax(1)==label).float().mean().item()*100)
            f1_list.append(f1_score(labs, preds, average='macro')*100)
            rec_list.append(recall_score(labs, preds, average='macro')*100)

    acc_m, acc_ci = compute_ci(acc_list)
    f1_m,  f1_ci  = compute_ci(f1_list)
    r_m,   r_ci   = compute_ci(rec_list)
    return acc_m, acc_ci, f1_m, f1_ci, r_m, r_ci


# ─────────────────────────────────────────
# 【修复后】Cross-Domain Eval，用correct_cd_split + 可选FT
# ─────────────────────────────────────────
def validate_cd(args, model, loader, n_ep=600, ft_steps=0):
    """
    v4: enhanced直接=feat(raw)，无需再区分raw/enhanced
    """
    model.eval()
    acc_list, f1_list, rec_list = [], [], []

    for i, (data, _) in enumerate(loader, 1):
        if i > n_ep: break
        data = data.cuda()
        support, query, query_label = correct_cd_split(
            data, args.shot, args.test_way, args.test_query)
        label = query_label.cuda()

        if ft_steps > 0:
            # FT涉及backward，不能用no_grad/autocast半精度
            logits = model.forward_eval_ft(
                support, query, args.shot, args.test_way, args.test_query,
                ft_steps=ft_steps, ft_lr=args.ft_lr)
        else:
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                logits = model.forward_eval(
                    support, query, args.shot, args.test_way, args.test_query)

        preds = logits.argmax(1).cpu().numpy()
        labs  = label.cpu().numpy()
        acc_list.append((logits.argmax(1)==label).float().mean().item()*100)
        f1_list.append(f1_score(labs, preds, average='macro')*100)
        rec_list.append(recall_score(labs, preds, average='macro')*100)

    acc_m, acc_ci = compute_ci(acc_list)
    f1_m,  f1_ci  = compute_ci(f1_list)
    r_m,   r_ci   = compute_ci(rec_list)
    return acc_m, acc_ci, f1_m, f1_ci, r_m, r_ci


def cross_domain_eval(args, model):
    print("\n" + "="*60)
    print(f"Cross-Domain Eval (v3, FIXED split, {args.shot}-shot, "
          f"600 episodes, ft_steps={args.ft_steps})")
    print("="*60)
    results = {}
    for name in ['eurosat', 'isic']:
        for shot_test in ([1, 5] if args.cd_both_shots else [args.shot]):
            args_shot_bak = args.shot
            args.shot = shot_test
            loader = get_cd_loader(name, args)
            if loader is None:
                args.shot = args_shot_bak
                continue
            acc_m, acc_ci, f1_m, _, r_m, _ = validate_cd(
                args, model, loader, 600, ft_steps=args.ft_steps)
            print(f"  {name:10s} {shot_test}-shot: {acc_m:.2f} ± {acc_ci:.2f}%  "
                  f"F1={f1_m:.2f}  R={r_m:.2f}")
            results[(name, shot_test)] = {'acc': acc_m, 'ci': acc_ci}
            args.shot = args_shot_bak
    torch.save(results, osp.join(args.save_path, f'cd_results_v3.pt'))
    return results


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main(args):
    seed_torch(args.seed)
    set_gpu(args.gpu)
    ensure_path(args.save_path)

    if args.dataset in ['mini','tiered']: args.size = 84
    elif args.dataset == 'cifarfs':       args.size = 32; args.worker = 0

    model = DDResNetEIRM(args).cuda()

    if args.pretrain_path and os.path.exists(args.pretrain_path):
        ckpt = torch.load(args.pretrain_path, map_location='cuda')
        missing, unexpected = model.encoder.load_state_dict(ckpt, strict=False)
        print(f"Loaded pretrained backbone: missing={len(missing)} unexpected={len(unexpected)}")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"DD-ResNet12-EIRM v6 (raw-only, +smooth-aug)  |  params={total_p:,}")
    print(f"配置: metric={args.metric} gate_mode={args.gate_mode} "
          f"style_aug={args.style_aug} ssl_coef={args.ssl_coef} "
          f"con_coef={args.con_coef} dis_weight={args.dis_weight} "
          f"irm_coef={args.irm_coef}")
    print(f"v6新增: smooth_aug={args.smooth_aug} "
          f"smooth_kernel={args.smooth_kernel_size} "
          f"smooth_alpha=({args.smooth_alpha_min},{args.smooth_alpha_max}) "
          f"smooth_prob={args.smooth_prob}")

    # Eval-only
    if args.max_epoch == 0:
        ckpt = osp.join(args.save_path, 'best.pth')
        if not osp.exists(ckpt):
            print(f"ERROR: {ckpt} not found"); return
        model.load_state_dict(torch.load(ckpt, map_location='cuda'))
        if args.cross_domain:
            cross_domain_eval(args, model)
        return

    train_loader = get_loader(args, 'train')
    val_loader   = get_loader(args, 'val' if args.dataset != 'cifarfs' else 'test')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch, eta_min=1e-6)
    scaler = GradScaler()

    best_acc, best_ep = 0., 0
    timer = Timer()

    print(f"\n{'─'*60}")
    print(f"Training: {args.dataset} | {args.shot}-shot | {args.max_epoch} epochs")
    print(f"train_batch={args.train_batch}  test_batch={args.test_batch}")
    print(f"IRM starts @ epoch {args.irm_start}, ramp={args.irm_ramp} epochs")
    print(f"{'─'*60}\n")

    for epoch in range(1, args.max_epoch + 1):
        tl, ta = train_epoch(args, model, train_loader, optimizer, scaler, epoch)
        scheduler.step()

        acc_m, acc_ci, f1_m, f1_ci, r_m, r_ci = validate(
            args, model, val_loader, args.test_batch)

        if acc_m > best_acc:
            best_acc = acc_m; best_ep = epoch
            torch.save(model.state_dict(), osp.join(args.save_path, 'best.pth'))
        torch.save(model.state_dict(), osp.join(args.save_path, 'last.pth'))

        w_irm = args.irm_coef * min(1., max(0., (epoch-args.irm_start)/args.irm_ramp))

        extra = f"w_irm={w_irm:.3f}"
        if args.metric == 'cosine':
            extra += f" cos_T={model.cosine.scale.item():.2f}"
        if args.gate_mode == 'soft':
            extra += f" gate_a={torch.sigmoid(model.encoder.style_gate.alpha).item():.3f}"

        print(f"Ep{epoch:03d}/{args.max_epoch} | "
              f"loss={tl:.4f} tr={ta:.4f} | "
              f"val={acc_m:.2f}±{acc_ci:.2f}% F1={f1_m:.2f} R={r_m:.2f} | "
              f"best={best_acc:.2f}%@ep{best_ep} | "
              f"{extra} | {timer.elapsed()}")

    print(f"\nDone. Best={best_acc:.2f}% @ epoch {best_ep}")
    print(f"Model: {args.save_path}/best.pth")

    if args.cross_domain:
        model.load_state_dict(torch.load(osp.join(args.save_path, 'best.pth'),
                                          map_location='cuda'))
        cross_domain_eval(args, model)


# ─────────────────────────────────────────
# Args
# ─────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--dataset',     default='mini', choices=['mini','tiered','cifarfs'])
    p.add_argument('--shot',        type=int, default=1)
    p.add_argument('--train-way',   type=int, default=5)
    p.add_argument('--test-way',    type=int, default=5)
    p.add_argument('--train-query', type=int, default=15)
    p.add_argument('--test-query',  type=int, default=15)
    p.add_argument('--train-batch', type=int, default=100)
    p.add_argument('--test-batch',  type=int, default=600)
    p.add_argument('--worker',      type=int, default=4)

    p.add_argument('--num-crops',   type=int,   default=2)
    p.add_argument('--dis-weight',  type=float, default=0.5,
                   help='Disentangle loss weight (消融v5c: 设为0关闭)')

    # ── 【消融实验】各组件开关 ──
    p.add_argument('--ssl-coef', type=float, default=0.5,
                   help='SSL(旋转)loss系数 (消融v5a: 设为0关闭)')
    p.add_argument('--con-coef', type=float, default=0.2,
                   help='Consistency(KL)loss系数 (消融v5b: 设为0关闭)')
    p.add_argument('--style-aug', dest='style_aug', action='store_true', default=True,
                   help='是否使用style augmentation (默认开启)')
    p.add_argument('--no-style-aug', dest='style_aug', action='store_false',
                   help='消融v5d: 关闭style augmentation '
                        '(x_style=x_original, L_fsl/L_dis/L_con退化)')
    p.add_argument('--gate-mode', default='soft',
                   choices=['soft', 'hard', 'identity'],
                   help='StyleNormGate模式: soft(可学习混合,默认) / '
                        'hard(纯IN,消融v5e) / identity(不归一化,消融v5f)')
    p.add_argument('--metric', default='cosine',
                   choices=['cosine', 'euclidean'],
                   help='ProtoNet距离度量 (消融v5g: euclidean还原v1行为)')

    # ── 【v6新增】平滑/低纹理对比度增强 ──
    p.add_argument('--smooth-aug', dest='smooth_aug', action='store_true', default=True,
                   help='是否使用smooth_style增强 (默认开启)')
    p.add_argument('--no-smooth-aug', dest='smooth_aug', action='store_false',
                   help='消融: 关闭smooth_style增强')
    p.add_argument('--smooth-kernel-size', type=int, default=5,
                   help='smooth_style的avg_pool kernel size')
    p.add_argument('--smooth-alpha-min', type=float, default=0.3,
                   help='smooth_style混合系数下界')
    p.add_argument('--smooth-alpha-max', type=float, default=0.7,
                   help='smooth_style混合系数上界')
    p.add_argument('--smooth-prob', type=float, default=0.5,
                   help='每个batch应用smooth_style的概率')

    p.add_argument('--irm-penalty', type=float, default=0.1)
    p.add_argument('--irm-coef',    type=float, default=0.3)
    p.add_argument('--irm-start',   type=int,   default=80)
    p.add_argument('--irm-ramp',    type=int,   default=30)

    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--wd',          type=float, default=5e-4)
    p.add_argument('--max-epoch',   type=int,   default=200)

    p.add_argument('--gpu',         default='0')
    p.add_argument('--seed',        type=int,   default=1337)
    p.add_argument('--save-path',   default='./save/dd_eirm_v6')

    p.add_argument('--pretrain-path', default='')

    p.add_argument('--cross-domain',  action='store_true')
    p.add_argument('--cd-both-shots', action='store_true',
                   help='cross-domain eval both 1-shot and 5-shot regardless of --shot')
    p.add_argument('--eurosat-path',  default='')
    p.add_argument('--isic-csv',      default='')
    p.add_argument('--isic-img-dir',  default='')

    # 改动5: Test-Time Fine-Tuning
    p.add_argument('--ft-steps', type=int, default=0,
                   help='test-time fine-tuning steps (0=disabled). '
                        'v4: only tunes cosine.scale (1 param)')
    p.add_argument('--ft-lr',    type=float, default=0.05)

    args = p.parse_args()
    t0 = datetime.datetime.now()
    main(args)
    print(f"Total: {datetime.datetime.now()-t0}")
