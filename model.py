"""CATFormer: Cloud-Aware Temporal fusion with SAR gating and uncertainty.

What is new vs. UnCRtainTS / U-TAE:
 1. Cloud-biased temporal attention: attention logits get  -beta_h * cloud(t, pixel), so each head
    learns how strongly to distrust cloudy observations, per pixel, at every scale.
 2. SAR gate: radar features are injected into the optical stream through a gate driven by the
    cloud map, so SAR fills in only where optics are occluded (SAR is cloud-penetrating but speckled).
 3. Residual-on-composite decoding: the network first forms an attention-weighted composite of the
    input frames at full resolution, then predicts a correction - it only has to learn what no
    input frame saw clearly.
 4. Per-pixel, per-band aleatoric uncertainty (Gaussian NLL), as in UnCRtainTS.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = nn.GroupNorm(1, c)
        self.pw = nn.Sequential(nn.Conv2d(c, 4 * c, 1), nn.GELU(), nn.Conv2d(4 * c, c, 1))

    def forward(self, x):
        return x + self.pw(self.norm(self.dw(x)))


class SARGatedStem(nn.Module):
    def __init__(self, c, s2_ch=13, s1_ch=2):
        super().__init__()
        self.opt = nn.Conv2d(s2_ch + 1, c, 3, padding=1)
        self.sar = nn.Sequential(nn.Conv2d(s1_ch, c, 3, padding=1), Block(c))
        self.gate = nn.Sequential(nn.Conv2d(2 * c + 1, c, 1), nn.Sigmoid())

    def forward(self, s2, s1, cloud):
        o = self.opt(torch.cat([s2, cloud], 1))
        s = self.sar(s1)
        g = self.gate(torch.cat([o, s, cloud], 1))
        return o + g * s


class CloudBiasedTemporalAttention(nn.Module):
    """Per-pixel attention across T with a learned master query (L-TAE style)."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.h, self.dk = heads, c // heads
        self.k = nn.Conv2d(c, c, 1)
        self.q = nn.Parameter(torch.randn(heads, self.dk) * 0.02)
        self.beta = nn.Parameter(torch.full((heads,), 4.0))
        self.v = nn.Conv2d(c, c, 1)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x, cloud, temb):
        # x: B,T,C,H,W  cloud: B,T,1,H,W  temb: B,T,C
        B, T, C, H, W = x.shape
        xf = (x + temb[..., None, None]).flatten(0, 1)
        k = self.k(xf).view(B, T, self.h, self.dk, H, W)
        logits = torch.einsum("bthdxy,hd->bthxy", k, self.q) / math.sqrt(self.dk)
        logits = logits - F.softplus(self.beta)[None, None, :, None, None] * cloud
        a = logits.softmax(1)  # B,T,h,H,W
        v = self.v(x.flatten(0, 1)).view(B, T, self.h, self.dk, H, W)
        y = (a[:, :, :, None] * v).sum(1).reshape(B, C, H, W)
        return self.out(y), a.mean(2)  # attention averaged over heads: B,T,H,W


def time_embedding(tpos, c):
    half = c // 2
    freq = torch.exp(-math.log(100.0) * torch.arange(half, device=tpos.device) / half)
    ang = tpos[..., None] * freq
    return torch.cat([ang.sin(), ang.cos()], -1)


class CATFormer(nn.Module):
    def __init__(self, s2_ch=13, widths=(48, 96, 192, 384), depth=2, heads=4):
        super().__init__()
        self.widths = widths
        self.stem = SARGatedStem(widths[0], s2_ch)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, c in enumerate(widths):
            self.enc.append(nn.Sequential(*[Block(c) for _ in range(depth)]))
            if i < len(widths) - 1:
                self.down.append(nn.Conv2d(c, widths[i + 1], 2, stride=2))
        self.tattn = nn.ModuleList([CloudBiasedTemporalAttention(c, heads) for c in widths])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(nn.Sequential(nn.Conv2d(2 * widths[i - 1], widths[i - 1], 1),
                                          *[Block(widths[i - 1]) for _ in range(depth)]))
        self.head = nn.Conv2d(widths[0] + s2_ch, 2 * s2_ch, 3, padding=1)

    def forward(self, s2, s1, cloud, tpos):
        B, T, C, H, W = s2.shape
        x = self.stem(s2.flatten(0, 1), s1.flatten(0, 1), cloud.flatten(0, 1))
        skips, cl = [], cloud
        for i, c in enumerate(self.widths):
            x = self.enc[i](x)
            agg, a = self.tattn[i](x.view(B, T, c, *x.shape[-2:]), cl, time_embedding(tpos, c))
            skips.append(agg)
            if i == 0:
                attn_full = a
            if i < len(self.widths) - 1:
                x = self.down[i](x)
                cl = F.avg_pool2d(cl.flatten(0, 1), 2).view(B, T, 1, *x.shape[-2:])
        y = skips[-1]
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            y = dec(torch.cat([up(y), skips[-2 - j]], 1))
        composite = (attn_full[:, :, None] * s2).sum(1)
        out = self.head(torch.cat([y, composite], 1))
        mean = (composite + out[:, :C]).clamp(0, 1)
        logvar = out[:, C:].clamp(-10, 5)
        return mean, logvar, attn_full


def ssim_loss(x, y, rgb=(3, 2, 1)):
    x, y = x[:, list(rgb)], y[:, list(rgb)]
    mu_x, mu_y = F.avg_pool2d(x, 7, 1, 3), F.avg_pool2d(y, 7, 1, 3)
    sxx = F.avg_pool2d(x * x, 7, 1, 3) - mu_x ** 2
    syy = F.avg_pool2d(y * y, 7, 1, 3) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, 7, 1, 3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return 1 - s.mean()


def sam_loss(x, y):
    cos = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + 1e-6)
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()


def loss_fn(mean, logvar, y, occl=None, w_nll=0.1, w_ssim=0.2, w_sam=0.1, w_occl=1.0):
    """L1 (up-weighted where every input frame was cloudy) + NLL + SSIM + SAM.
    occl: B,1,H,W = min over inputs of cloud prob, i.e. how hidden the pixel was in all frames."""
    err = (mean - y).abs()
    w = 1 + w_occl * occl if occl is not None else 1
    l1 = (err * w).mean() / (w.mean() if occl is not None else 1)
    nll = 0.5 * (logvar + (mean.detach() - y) ** 2 * torch.exp(-logvar)).mean()
    loss = l1 + w_nll * nll + w_ssim * ssim_loss(mean, y) + w_sam * sam_loss(mean, y)
    return loss, l1.detach(), nll.detach()


class Refiner(nn.Module):
    """Stage 2: rectified-flow bridge from the stage-1 estimate m to the clear image y.
    x_s = (1-s) m + s y + sigma*sqrt(s(1-s)) eps, predict v = y - m. Conditioned on m, the
    uncertainty map, the composite-free inputs' cloud occlusion and SAR, so it spends its
    capacity where stage 1 was unsure."""

    def __init__(self, ch=13, cond=13 + 13 + 1 + 2, w=64, depth=2):
        super().__init__()
        self.inp = nn.Conv2d(ch + cond, w, 3, padding=1)
        self.temb = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w))
        self.e1 = nn.Sequential(*[Block(w) for _ in range(depth)])
        self.d1 = nn.Conv2d(w, 2 * w, 2, stride=2)
        self.e2 = nn.Sequential(*[Block(2 * w) for _ in range(depth)])
        self.d2 = nn.Conv2d(2 * w, 4 * w, 2, stride=2)
        self.mid = nn.Sequential(*[Block(4 * w) for _ in range(2 * depth)])
        self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.f2 = nn.Sequential(nn.Conv2d(4 * w, 2 * w, 1), *[Block(2 * w) for _ in range(depth)])
        self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.f1 = nn.Sequential(nn.Conv2d(2 * w, w, 1), *[Block(w) for _ in range(depth)])
        self.out = nn.Conv2d(w, ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.w = w

    def forward(self, x, s, cond):
        h = self.inp(torch.cat([x, cond], 1))
        h = h + self.temb(time_embedding(s * 10, self.w))[..., None, None]
        a = self.e1(h); b = self.e2(self.d1(a)); c = self.mid(self.d2(b))
        b = self.f2(torch.cat([self.u2(c), b], 1))
        a = self.f1(torch.cat([self.u1(b), a], 1))
        return self.out(a)

    @staticmethod
    def make_cond(m, logvar, occl, s1_mean):
        return torch.cat([m, logvar / 5, occl, s1_mean], 1)

    def train_loss(self, m, y, cond, sigma=0.02):
        s = torch.rand(m.shape[0], device=m.device)
        sv = s[:, None, None, None]
        x = (1 - sv) * m + sv * y + sigma * (sv * (1 - sv)).sqrt() * torch.randn_like(m)
        return (self(x, s, cond) - (y - m)).abs().mean()

    @torch.no_grad()
    def sample(self, m, cond, steps=4):
        x = m
        for i in range(steps):
            s = torch.full((m.shape[0],), i / steps, device=m.device)
            x = x + self(x, s, cond) / steps
        return x.clamp(0, 1)
