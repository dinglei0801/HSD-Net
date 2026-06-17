"""
models/dd_resnet12_v6.py — v5基础上新增"平滑/低纹理对比度增强"

相对dd_resnet12_v5.py的改动:
  - StyleAugmentation新增 smooth_style() 静态方法
    动机: diagnose_domain_shift.py显示 GLCM对比度
          miniImageNet≈1.89 远高于 EuroSAT≈0.33 / ISIC≈0.29，
          即目标域图像局部纹理远比训练分布"平滑"。
          用average pooling模糊+混合，让模型训练时也见到
          "平滑版"图像，缩小这一统计量上的训练/目标域差距。
  - DDResNetEIRM新增配置: --smooth-aug(默认开) / --smooth-kernel-size
    / --smooth-alpha-min / --smooth-alpha-max / --smooth-prob
    在forward_train的style分支里，以smooth_prob概率对
    x_shot_style/x_query_style叠加smooth_style

其余结构(包括v5的gate_mode/metric/ssl_coef/con_coef/style_aug等
全部消融开关)与dd_resnet12_v5.py完全一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import random


# ════════════════════════════════════════════
# 基础组件（与dd_resnet12.py一致）
# ════════════════════════════════════════════

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        if not self.training:
            return x
        B, C, H, W = x.shape
        mask = Bernoulli(gamma).sample(
            (B, C, H - self.block_size + 1, W - self.block_size + 1)
        ).to(x.device)
        lp = (self.block_size - 1) // 2
        rp = self.block_size // 2
        padded = F.pad(mask, (lp, rp, lp, rp))
        nz = mask.nonzero()
        if nz.shape[0] > 0:
            off = torch.stack([
                torch.arange(self.block_size).view(-1,1)
                    .expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size)
            ]).t().to(x.device)
            off = torch.cat([torch.zeros(self.block_size**2, 2, dtype=torch.long,
                                          device=x.device), off.long()], 1)
            nz2  = nz.repeat(self.block_size**2, 1)
            off2 = off.repeat(nz.shape[0], 1).view(-1, 4)
            idx  = nz2 + off2
            padded[idx[:,0], idx[:,1], idx[:,2], idx[:,3]] = 1.
        block_mask = 1 - padded
        cnt  = block_mask.numel()
        ones = block_mask.sum().clamp(min=1)
        return block_mask * x * (cnt / ones)


class BasicBlock(nn.Module):
    def __init__(self, inp, planes, stride=1,
                 drop_rate=0., drop_block=False, block_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp,    planes, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes)
        self.relu  = nn.LeakyReLU(0.1)
        self.pool  = nn.MaxPool2d(stride)
        self.ds    = nn.Sequential(
            nn.Conv2d(inp, planes, 1, bias=False),
            nn.BatchNorm2d(planes)
        ) if inp != planes else None
        self.drop_rate  = drop_rate
        self.drop_block = drop_block
        self.DB = DropBlock(block_size)
        self.n_batch = 0

    def forward(self, x):
        self.n_batch += 1
        res = x if self.ds is None else self.ds(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + res)
        out = self.pool(out)
        if self.drop_rate > 0:
            if self.drop_block:
                fs    = out.size(2)
                keep  = max(1. - self.drop_rate / (20*2000) * self.n_batch,
                            1. - self.drop_rate)
                gamma = (1-keep)/self.DB.block_size**2 * fs**2 / (fs-self.DB.block_size+1)**2
                out   = self.DB(out, gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


# ════════════════════════════════════════════
# 改动3: StyleNormGate -> 可学习软混合
# ════════════════════════════════════════════

class StyleNormalizationGate(nn.Module):
    """
    x_out = a * (gamma*IN(x)+beta) + (1-a) * x

    mode='soft'(默认):  a = sigmoid(alpha), alpha可学习, init alpha=1.0 -> a≈0.73
                        模型自行决定IN的混合比例

    mode='hard':        a恒为1.0 (纯Instance Norm, 与v1的硬编码行为一致)
                        gamma/beta仍可学习，但混合比例不可学习/不可变

    mode='identity':    a恒为0.0 (完全不做归一化，StyleNormGate=Identity)
                        gamma/beta不起作用

    用于消融实验: 'soft' vs 'hard' vs 'identity' 三种模式
    对比StyleNormGate本身的贡献和"可学习混合"这一设计的贡献
    """
    def __init__(self, channels=160, mode='soft'):
        super().__init__()
        assert mode in ('soft', 'hard', 'identity')
        self.mode  = mode
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        if mode == 'soft':
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            # hard/identity: alpha不可学习，注册为buffer方便state_dict兼容
            self.register_buffer('alpha', torch.tensor(
                10.0 if mode == 'hard' else -10.0))  # sigmoid(±10)≈1/0

    def forward(self, x):
        if self.mode == 'identity':
            return x
        mu    = x.mean(dim=[2, 3], keepdim=True)
        sigma = x.var(dim=[2, 3], keepdim=True).add(1e-5).sqrt()
        x_norm = self.gamma * (x - mu) / sigma + self.beta
        if self.mode == 'hard':
            return x_norm
        a = torch.sigmoid(self.alpha)
        return a * x_norm + (1 - a) * x


# ════════════════════════════════════════════
# 【v4改动】RelationModule已移除
#
# 原因: raw特征在所有已测试模型上都比经过RelationModule的
# enhanced特征好3-9%。直接去掉这个模块。
# ════════════════════════════════════════════


# ════════════════════════════════════════════
# 改动2: 可学习温度的Cosine ProtoNet
# ════════════════════════════════════════════

class CosineProto(nn.Module):
    """
    logits = scale * cosine_similarity(query, proto)
    scale 可学习，初始10.0（常见设置）
    """
    def __init__(self, init_scale=10.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, query, proto):
        q = F.normalize(query, dim=-1)
        p = F.normalize(proto, dim=-1)
        return self.scale * (q @ p.t())


# ════════════════════════════════════════════
# 主体: DDResNet12
# ════════════════════════════════════════════

class EuclideanProto(nn.Module):
    """
    logits = -||query - proto||^2
    无参数，对应v1的原始距离方式（用于消融实验 --metric euclidean）
    """
    def forward(self, query, proto):
        n, m = query.size(0), proto.size(0)
        a = query.unsqueeze(1).expand(n, m, -1)
        b = proto.unsqueeze(0).expand(n, m, -1)
        return -((a - b) ** 2).sum(dim=2)


class DDResNet12(nn.Module):
    def __init__(self, drop_rate=0.1, dropblock_size=5, gate_mode='soft'):
        super().__init__()
        self.layer1 = BasicBlock(3,   64,  stride=2, drop_rate=drop_rate)
        self.layer2 = BasicBlock(64,  160, stride=2, drop_rate=drop_rate)
        self.style_gate = StyleNormalizationGate(channels=160, mode=gate_mode)

        self.layer3 = BasicBlock(160, 320, stride=2, drop_rate=drop_rate,
                                 drop_block=True, block_size=dropblock_size)
        self.layer4 = BasicBlock(320, 640, stride=2, drop_rate=drop_rate,
                                 drop_block=True, block_size=dropblock_size)

        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        # self.relation removed in v4 (raw feature only)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        shallow = h
        h = self.style_gate(h)

        h = self.layer3(h)
        h = self.layer4(h)

        h = self.avgpool(h)
        feat     = h.view(h.size(0), -1)
        enhanced = feat   # v4: 直接用raw特征，不经过RelationModule

        return feat, enhanced, shallow


# ════════════════════════════════════════════
# Style Augmentation（改动4: 新增极端色彩环境）
# ════════════════════════════════════════════

class StyleAugmentation:
    @staticmethod
    def random_style_transfer(x, alpha=0.5):
        B, C, H, W = x.shape
        if B < 2:
            return x
        perm = torch.randperm(B, device=x.device)
        ref  = x[perm]
        mu_src  = x.mean(dim=[2,3], keepdim=True)
        std_src = x.var(dim=[2,3],  keepdim=True).add(1e-5).sqrt()
        mu_ref  = ref.mean(dim=[2,3], keepdim=True)
        std_ref = ref.var(dim=[2,3],  keepdim=True).add(1e-5).sqrt()
        x_transferred = (x - mu_src) / std_src * std_ref + mu_ref
        x_out = alpha * x_transferred + (1 - alpha) * x
        return torch.clamp(x_out, 0, 1)

    @staticmethod
    def crop_style_transfer(x, k=2, scale=(0.2, 0.4)):
        B, C, H, W = x.shape
        result = x.clone()
        for _ in range(k):
            s = random.uniform(scale[0], scale[1])
            ch = max(1, int(H * s))
            cw = max(1, int(W * s))
            top  = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            crop = x[:, :, top:top+ch, left:left+cw]
            mu_crop  = crop.mean(dim=[2,3], keepdim=True)
            std_crop = crop.var(dim=[2,3],  keepdim=True).add(1e-5).sqrt()
            mu_x  = result.mean(dim=[2,3], keepdim=True)
            std_x = result.var(dim=[2,3],  keepdim=True).add(1e-5).sqrt()
            result = (result - mu_x) / std_x * std_crop + mu_crop
            result = torch.clamp(result, 0, 1)
        return result

    @staticmethod
    def smooth_style(x, kernel_size=5, alpha_range=(0.3, 0.7)):
        """
        【v6新增】平滑/低纹理对比度增强

        诊断数据(diagnose_domain_shift.py)显示:
          GLCM对比度: miniImageNet≈1.89  vs  EuroSAT≈0.33  ISIC≈0.29
        即EuroSAT/ISIC的图像局部纹理远比miniImageNet"平滑"。
        训练时模型几乎只见过mini这种高对比度纹理，
        这个增强让模型在训练阶段也见到"更平滑"的版本，
        缩小训练分布与低纹理对比度目标域之间的差距。

        实现: 用average pooling模糊图像，再与原图按alpha混合。
        kernel_size越大/alpha越大 -> 越平滑。
        """
        alpha = random.uniform(*alpha_range)
        blurred = F.avg_pool2d(x, kernel_size, stride=1,
                               padding=kernel_size // 2,
                               count_include_pad=False)
        out = alpha * blurred + (1 - alpha) * x
        return torch.clamp(out, 0, 1)

    @staticmethod
    def extreme_color_shift(x):
        """
        【新增】模拟EuroSAT/ISIC级别的极端色彩偏移:
          1. 转灰度后按随机权重重新着色（破坏色彩-语义的强绑定）
          2. 大幅度channel-wise亮度/对比度扰动

        EuroSAT是卫星图(植被=绿/水体=蓝/建筑=灰)
        ISIC是皮肤图像(肤色调为主)
        miniImageNet的色彩-语义关联与这两者都差异巨大

        这个环境强制模型在"色彩信息被打乱"时仍能基于形状/纹理分类
        """
        B, C, H, W = x.shape
        # 转灰度 (ITU-R 601-2 luma)
        gray = (0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3])
        # 随机重新分配到3个channel，权重随机（每个batch一组随机权重）
        w = torch.rand(3, device=x.device)
        w = w / w.sum()
        recolored = torch.cat([gray*w[0]*3, gray*w[1]*3, gray*w[2]*3], dim=1)
        # 加大幅度的channel-wise亮度扰动
        shift = (torch.rand(B, C, 1, 1, device=x.device) - 0.5) * 0.6
        out = recolored + shift
        return torch.clamp(out, 0, 1)

    @staticmethod
    def generate_environments(x):
        """
        5个环境:
          0: Original
          1: H-Flip
          2: Random Style Transfer (中等强度)
          3: Crop Style Transfer (SVasP核心)
          4: Extreme Color Shift (新增，模拟大domain gap)
        """
        envs = [
            x,
            torch.flip(x, dims=[3]),
            StyleAugmentation.random_style_transfer(x, alpha=0.7),
            StyleAugmentation.crop_style_transfer(x, k=2),
            StyleAugmentation.extreme_color_shift(x),
        ]
        return envs


# ════════════════════════════════════════════
# DisentangleLoss（与原版一致）
# ════════════════════════════════════════════

class DisentangleLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, shallow_orig, shallow_style):
        mu_o    = shallow_orig.mean(dim=[2,3])
        std_o   = shallow_orig.var(dim=[2,3]).add(1e-5).sqrt()
        mu_s    = shallow_style.mean(dim=[2,3])
        std_s   = shallow_style.var(dim=[2,3]).add(1e-5).sqrt()
        loss_mu  = F.mse_loss(mu_o,  mu_s)
        loss_std = F.mse_loss(std_o, std_s)
        return self.weight * (loss_mu + loss_std)


# ════════════════════════════════════════════
# IRM Loss（与原版一致，但内部logits现在是cosine尺度）
# ════════════════════════════════════════════

class IRMLoss(nn.Module):
    def __init__(self, n_envs=5, penalty_weight=0.1):
        super().__init__()
        self.n_envs         = n_envs
        self.penalty_weight = penalty_weight
        self.env_weights    = nn.Parameter(torch.ones(n_envs))

    def forward(self, logits, labels, env_ids):
        w    = F.softmax(self.env_weights, dim=0)
        loss = torch.tensor(0., device=logits.device)
        pen  = torch.tensor(0., device=logits.device)
        for e in range(self.n_envs):
            mask = (env_ids == e)
            if not mask.any():
                continue
            el, ey = logits[mask], labels[mask]
            scale  = torch.ones(1, requires_grad=True, device=logits.device)
            el_s   = F.cross_entropy(el * scale, ey)
            grad   = torch.autograd.grad(el_s, scale, create_graph=True)[0]
            loss   = loss + w[e] * F.cross_entropy(el, ey)
            pen    = pen  + w[e] * grad.pow(2)
        entropy = -(w * (w + 1e-8).log()).sum()
        return loss + self.penalty_weight * pen - 0.01 * entropy


# ════════════════════════════════════════════
# 完整模型
# ════════════════════════════════════════════

class DDResNetEIRM(nn.Module):
    def __init__(self, args):
        super().__init__()
        dropblock = 5 if args.dataset in ['mini','tiered','cub',
                                           'cars','places','plantae',
                                           'chestx','isic','eurosat',
                                           'cropdisease'] else 2
        gate_mode = getattr(args, 'gate_mode', 'soft')
        metric    = getattr(args, 'metric', 'cosine')

        self.encoder      = DDResNet12(drop_rate=0.1, dropblock_size=dropblock,
                                       gate_mode=gate_mode)
        self.style_aug    = StyleAugmentation()
        self.disentangle  = DisentangleLoss(weight=args.dis_weight)
        self.irm_loss     = IRMLoss(n_envs=5, penalty_weight=args.irm_penalty)

        # 【消融】距离度量: cosine(可学习温度) vs euclidean(无参数)
        if metric == 'cosine':
            self.cosine = CosineProto(init_scale=10.0)
        else:
            self.cosine = EuclideanProto()
        self.metric = metric

        self.num_crops    = args.num_crops
        self.irm_start    = args.irm_start
        self.irm_ramp     = args.irm_ramp
        self.irm_coef     = args.irm_coef

        # 【消融】各loss项的系数，0表示关闭
        self.ssl_coef  = getattr(args, 'ssl_coef', 0.5)
        self.con_coef  = getattr(args, 'con_coef', 0.2)
        # 【消融】是否使用style augmentation (crop_style_transfer)
        self.use_style_aug = getattr(args, 'style_aug', True)

        # 【v6新增】平滑/低纹理对比度增强 (针对EuroSAT/ISIC的GLCM对比度
        # 远低于miniImageNet这一诊断发现)
        self.smooth_aug         = getattr(args, 'smooth_aug', True)
        self.smooth_kernel_size = getattr(args, 'smooth_kernel_size', 5)
        self.smooth_alpha_range = (getattr(args, 'smooth_alpha_min', 0.3),
                                    getattr(args, 'smooth_alpha_max', 0.7))
        self.smooth_prob        = getattr(args, 'smooth_prob', 0.5)

    def proto_logits(self, feat, shot, way, query_feat):
        proto = feat.reshape(shot, way, -1).mean(0)
        return self.cosine(query_feat, proto)

    def ssl_loss(self, data_shot, shot, way):
        rots  = torch.cat([
            data_shot.transpose(2,3).flip(2),
            data_shot.flip(2).flip(3),
            data_shot.flip(2).transpose(2,3)
        ], dim=0)
        _, enh_rot, _ = self.encoder(rots)
        _, enh_s,   _ = self.encoder(data_shot)
        proto  = enh_s.reshape(shot, way, -1).mean(0)
        label  = torch.arange(way).repeat(3 * shot).to(data_shot.device)
        logits = self.proto_logits(enh_s, shot, way, enh_rot)
        return F.cross_entropy(logits, label)

    def forward_train(self, data_shot, data_query, shot, way, n_query, epoch):
        device = data_shot.device
        label_q = torch.arange(way).repeat(n_query).to(device)

        feat_s, enh_s, shallow_s = self.encoder(data_shot)
        feat_q, enh_q, shallow_q = self.encoder(data_query)

        proto   = enh_s.reshape(shot, way, -1).mean(0)
        logits  = self.proto_logits(enh_s, shot, way, enh_q)
        L_cls   = F.cross_entropy(logits, label_q)

        # 【消融】是否使用style augmentation
        if self.use_style_aug:
            x_shot_style  = self.style_aug.crop_style_transfer(data_shot,  k=self.num_crops)
            x_query_style = self.style_aug.crop_style_transfer(data_query, k=self.num_crops)
        else:
            # 不做style扰动: x_style = x_original
            # L_fsl与L_cls等价, L_dis=0(shallow==shallow_sty), L_con≈0
            x_shot_style, x_query_style = data_shot, data_query

        # 【v6新增】以smooth_prob概率叠加平滑增强 (与crop_style_transfer
        # 独立, 即使--no-style-aug也可以单独生效, 让model见到"平滑版原图")
        if self.smooth_aug and random.random() < self.smooth_prob:
            x_shot_style  = self.style_aug.smooth_style(
                x_shot_style,  self.smooth_kernel_size, self.smooth_alpha_range)
            x_query_style = self.style_aug.smooth_style(
                x_query_style, self.smooth_kernel_size, self.smooth_alpha_range)

        _, enh_s_sty, shallow_s_sty = self.encoder(x_shot_style)
        _, enh_q_sty, shallow_q_sty = self.encoder(x_query_style)

        proto_sty  = enh_s_sty.reshape(shot, way, -1).mean(0)
        logits_sty = self.proto_logits(enh_s_sty, shot, way, enh_q_sty)
        L_fsl      = F.cross_entropy(logits_sty, label_q)

        L_dis_s = self.disentangle(shallow_s, shallow_s_sty)
        L_dis_q = self.disentangle(shallow_q, shallow_q_sty)
        L_dis   = (L_dis_s + L_dis_q) * 0.5

        # 【消融】SSL loss系数, 0表示关闭
        L_ssl = self.ssl_loss(data_shot, shot, way) if self.ssl_coef > 0 \
                else torch.tensor(0., device=device)

        # 【消融】Consistency loss系数, 0表示关闭
        if self.con_coef > 0:
            L_con = F.kl_div(
                F.log_softmax(logits_sty, dim=1),
                F.softmax(logits.detach(), dim=1),
                reduction='batchmean'
            )
        else:
            L_con = torch.tensor(0., device=device)

        w_irm = self.irm_coef * min(1., max(0.,
            (epoch - self.irm_start) / self.irm_ramp))
        L_eirm = torch.tensor(0., device=device)
        if w_irm > 0:
            envs    = self.style_aug.generate_environments(data_query)  # 现在是5个
            env_feats, env_ids = [], []
            for i, env_x in enumerate(envs):
                _, ef, _ = self.encoder(env_x)
                env_feats.append(ef)
                env_ids.append(torch.full((ef.size(0),), i,
                                          dtype=torch.long, device=device))
            all_feats = torch.cat(env_feats, dim=0)
            all_ids   = torch.cat(env_ids,   dim=0)
            all_label = label_q.repeat(len(envs))
            all_logits = self.proto_logits(enh_s.detach(), shot, way, all_feats)
            L_eirm = self.irm_loss(all_logits, all_label, all_ids)

        total = (L_cls
                 + L_fsl
                 + self.ssl_coef * L_ssl
                 + L_dis
                 + self.con_coef * L_con
                 + w_irm * L_eirm)

        acc = (logits.argmax(1) == label_q).float().mean().item()
        return total, L_cls, L_dis, L_eirm, acc, w_irm

    @torch.no_grad()
    def forward_eval(self, data_shot, data_query, shot, way, n_query):
        _, enh_s, _ = self.encoder(data_shot)
        _, enh_q, _ = self.encoder(data_query)
        return self.proto_logits(enh_s, shot, way, enh_q)

    # ── Test-Time Fine-Tuning (改动5，独立可选) ──
    def forward_eval_ft(self, data_shot, data_query, shot, way, n_query,
                         ft_steps=5, ft_lr=0.01):
        """
        FT只调 cosine.scale 这1个标量。
        【消融】euclidean模式下EuclideanProto没有可学习参数，
        FT没有目标，直接退化为普通forward_eval。
        """
        if self.metric != 'cosine' or ft_steps == 0:
            return self.forward_eval(data_shot, data_query, shot, way, n_query)

        orig_cosine_scale = self.cosine.scale.data.clone()

        ft_params = [self.cosine.scale]
        for p in ft_params:
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(ft_params, lr=ft_lr)

        label_s = torch.arange(way).to(data_shot.device)  # shot=1时每class一个

        for _ in range(ft_steps):
            _, enh_s, _ = self.encoder(data_shot)
            # leave-one-out: 用support自身做(shot-1)-shot分类loss
            if shot > 1:
                proto = enh_s.reshape(shot, way, -1).mean(0)
                logits_s = self.cosine(enh_s, proto)
                label_rep = torch.arange(way).repeat(shot).to(data_shot.device)
                loss = F.cross_entropy(logits_s, label_rep)
            else:
                # 1-shot时无法leave-one-out，用支撑集自重构loss(让scale适配新域统计量)
                proto = enh_s  # (way, D)
                logits_s = self.cosine(enh_s, proto)
                loss = F.cross_entropy(logits_s, label_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, enh_s, _ = self.encoder(data_shot)
            _, enh_q, _ = self.encoder(data_query)
            logits = self.proto_logits(enh_s, shot, way, enh_q)

        # 恢复原始参数（不污染下一个episode）
        self.cosine.scale.data.copy_(orig_cosine_scale)
        for p in ft_params:
            p.requires_grad_(False)

        return logits
