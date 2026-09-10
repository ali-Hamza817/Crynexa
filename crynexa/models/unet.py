"""Attacker architectures.

ResUNet is the primary attacker: an image-to-image inverse with skip
connections. PlainCNN is the no-skip encoder-decoder baseline.

Note the inductive-bias caveat: convolution assumes spatial locality, which a
full 2-D permutation destroys. Attack success is therefore a joint property of
cipher and attacker, not of the cipher alone. Measured, the effect is a
difference of degree rather than of kind: on the key-independent permutation
control ResUNet and LinearMixer both reach top-1 1.000, at +13.9 dB and
+15.5 dB over the prior floor respectively. Every cipher is attacked with more
than one architecture so that a resistance claim is made against the strongest
attacker tried.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.silu(h + self.skip(x))


class ResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, depth=3, logits=False):
        super().__init__()
        self.depth = depth
        self.logits = logits
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.stem = ResBlock(in_ch, chs[0])
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(nn.ModuleList([
                ResBlock(chs[i], chs[i + 1]),
                nn.Conv2d(chs[i + 1], chs[i + 1], 3, stride=2, padding=1),
            ]))
        self.mid = nn.Sequential(ResBlock(chs[-1], chs[-1]), ResBlock(chs[-1], chs[-1]))
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(chs[i + 1], chs[i + 1], 4, stride=2, padding=1),
                ResBlock(chs[i + 1] * 2, chs[i]),
            ]))
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        h = self.stem(x)
        skips = []
        for blk, down in self.downs:
            h = blk(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for (up, blk), s in zip(self.ups, reversed(skips)):
            h = up(h)
            h = blk(torch.cat([h, s], dim=1))
        h = self.head(h)
        return h if self.logits else torch.sigmoid(h)


class PlainCNN(nn.Module):
    """Encoder-decoder with no skip connections."""

    def __init__(self, in_ch=3, out_ch=3, base=64, depth=3, logits=False):
        super().__init__()
        self.logits = logits
        chs = [base * (2 ** i) for i in range(depth + 1)]
        enc = [ResBlock(in_ch, chs[0])]
        for i in range(depth):
            enc += [ResBlock(chs[i], chs[i + 1]),
                    nn.Conv2d(chs[i + 1], chs[i + 1], 3, stride=2, padding=1)]
        dec = []
        for i in reversed(range(depth)):
            dec += [nn.ConvTranspose2d(chs[i + 1], chs[i + 1], 4, stride=2, padding=1),
                    ResBlock(chs[i + 1], chs[i])]
        self.enc = nn.Sequential(*enc)
        self.dec = nn.Sequential(*dec)
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        h = self.head(self.dec(self.enc(x)))
        return h if self.logits else torch.sigmoid(h)


class PixelMixer(nn.Module):
    """Permutation-tolerant attacker: global token mixing over patch tokens.

    Convolution is the wrong inductive bias for permutation-dominant ciphers
    because scrambling destroys locality. This model mixes globally instead,
    so it can in principle re-assemble displaced content.
    """

    def __init__(self, in_ch=3, out_ch=3, img=32, patch=4, dim=384, depth=6, heads=6,
                 logits=False):
        super().__init__()
        self.logits = logits
        self.img, self.patch = img, patch
        self.gh = img // patch
        n_tok = self.gh * self.gh
        pdim = in_ch * patch * patch
        self.proj = nn.Linear(pdim, dim)
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_ch * patch * patch)
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        p, gh = self.patch, self.gh
        t = x.reshape(B, C, gh, p, gh, p).permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gh, -1)
        t = self.proj(t) + self.pos
        t = self.norm(self.enc(t))
        t = self.head(t).reshape(B, gh, gh, self.out_ch, p, p)
        t = t.permute(0, 3, 1, 4, 2, 5).reshape(B, self.out_ch, H, W)
        return t if self.logits else torch.sigmoid(t)


class LinearMixer(nn.Module):
    """Global learned linear un-mixing over the flattened image.

    A full 2-D pixel permutation is a global gather: output pixel i comes from
    an arbitrary source position. A dense N x N map represents an arbitrary
    fixed permutation exactly, with no locality assumption, so it is the
    natural attacker for permutation-dominant ciphers and reaches a higher
    gain over floor than the U-Net on the key-independent control
    (+15.5 dB vs +13.9 dB).
    """

    def __init__(self, in_ch=3, out_ch=3, img=32, hidden=0, refine=True, logits=False):
        super().__init__()
        self.logits = logits
        self.img, self.out_ch = img, out_ch
        n_in = in_ch * img * img
        n_out = out_ch * img * img
        self.mix = (nn.Linear(n_in, n_out) if not hidden else
                    nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(),
                                  nn.Linear(hidden, n_out)))
        self.refine = nn.Sequential(ResBlock(out_ch, 32), ResBlock(32, 32),
                                    nn.Conv2d(32, out_ch, 1)) if refine else None

    def forward(self, x):
        B = x.shape[0]
        h = self.mix(x.flatten(1)).reshape(B, self.out_ch, self.img, self.img)
        if self.refine is not None:
            h = h + self.refine(h)
        return h if self.logits else torch.sigmoid(h)


class WithPosEmbed(nn.Module):
    """Adds a free learnable per-position embedding to the input.

    A convolutional attacker is translation-equivariant, and smooth (x, y)
    coordinate channels are a poor basis for a pseudorandom position-dependent
    keystream. This gives the attacker the capacity to represent one. It grants
    no key information: under many keys there is no fixed keystream to memorise.
    """

    def __init__(self, net, ch, h, w):
        super().__init__()
        self.net = net
        self.pos = nn.Parameter(torch.zeros(1, ch, h, w))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        p = self.pos.expand(x.shape[0], -1, -1, -1).to(x.dtype)
        return self.net(torch.cat([x, p], 1))


def build_model(name, img_size=32, pos_embed=32, **kw):
    if pos_embed:
        kw = dict(kw)
        kw["in_ch"] = kw.get("in_ch", 3) + pos_embed
        net = _build_core(name, img_size, **kw)
        return WithPosEmbed(net, pos_embed, img_size, img_size)
    return _build_core(name, img_size, **kw)


def _build_core(name, img_size=32, **kw):
    if name == "resunet":
        return ResUNet(**kw)
    if name == "cnn":
        return PlainCNN(**kw)
    if name == "pixelmixer":
        return PixelMixer(img=img_size, **kw)
    if name == "linearmixer":
        return LinearMixer(img=img_size, **kw)
    raise ValueError(name)
