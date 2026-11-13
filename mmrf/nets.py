"""Networks: the teacher (3D CNN encoder E, per-position token encoder g, token-conditioned FiLM decoder), the spatial transformer, and the student."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LEVEL_CHANNELS = (96, 192, 384)                                              # encoder levels 1 / 2 / 3 at 24^3, 12^3, 6^3


# ---------------------------------------------------------------- teacher: encoder and tokens
def _block(cin, cout, stride):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride, 1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.2, True),
                         nn.Conv3d(cout, cout, 3, 1, 1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.2, True))


class CNNEncoder(nn.Module):
    """E: (T, 1, 96, 96, 96) -> {1: (T, 96, 24^3), 2: (T, 192, 12^3), 3: (T, 384, 6^3)}"""
    def __init__(self, ch=(32, 96, 192, 384)):
        super().__init__()
        self.b0 = _block(1, ch[0], 2); self.b1 = _block(ch[0], ch[1], 2); self.b2 = _block(ch[1], ch[2], 2); self.b3 = _block(ch[2], ch[3], 2)

    def forward(self, v):
        x = self.b0(v); l1 = self.b1(x); l2 = self.b2(l1); l3 = self.b3(l2); return {1: l1, 2: l2, 3: l3}


class TokenProjection(nn.Module):
    """g: per-position two-layer perceptron (672 -> 2 d -> d, GELU, layer norm) shared by all positions, plus a learned position embedding"""
    def __init__(self, token_dim=128, n_tokens=216, in_ch=sum(LEVEL_CHANNELS)):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_ch, 2 * token_dim), nn.GELU(), nn.Linear(2 * token_dim, token_dim), nn.LayerNorm(token_dim))
        self.spatial_pos = nn.Embedding(n_tokens, token_dim); self.register_buffer("pos_idx", torch.arange(n_tokens))

    def forward(self, x):                                                     # (T, N, 672) -> (T, N, d)
        return self.proj(x) + self.spatial_pos(self.pos_idx)


class MotionEncoder(nn.Module):
    """E followed by g: every level pooled to grid^3 positions, the frame-0 features subtracted (reference subtraction), channels concatenated
    (672), then g. tokens(feats)[t] is the motion token grid of frame t, (grid^3, d); tokens[0] encodes no motion."""
    def __init__(self, token_dim=128, grid=6):
        super().__init__(); self.grid = grid
        self.feat_enc = CNNEncoder(); self.frame_encoder = TokenProjection(token_dim, grid ** 3)

    def features(self, vols):                                                 # (T, 1, 96, 96, 96) -> {level: (T, C, ...)}
        return self.feat_enc(vols)

    def tokens(self, feats):
        parts = []
        for l in (1, 2, 3):
            h = F.adaptive_avg_pool3d(feats[l], self.grid); parts.append((h - h[0:1].detach()).flatten(2).permute(0, 2, 1))
        return self.frame_encoder(torch.cat(parts, -1))


# ---------------------------------------------------------------- teacher: decoder
class FiLM(nn.Module):
    """per-channel scale and shift from a vector z"""
    def __init__(self, z_dim, ch):
        super().__init__(); self.fc = nn.Linear(z_dim, 2 * ch)

    def forward(self, x, z):
        g, b = self.fc(z).chunk(2, dim=1); return g[..., None, None, None] * x + b[..., None, None, None]


class SpatialFiLM(nn.Module):
    """per-voxel scale and shift: 1x1x1 convolution of the token grid trilinearly upsampled to the layer's resolution (identity at initialisation)"""
    def __init__(self, tok, ch):
        super().__init__(); self.conv = nn.Conv3d(tok, 2 * ch, 1); nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)
        with torch.no_grad(): self.conv.bias[:ch] = 1.0

    def forward(self, x, grid):
        g, b = self.conv(F.interpolate(grid, size=x.shape[-3:], mode="trilinear", align_corners=False)).chunk(2, dim=1); return g * x + b


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, z_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1); self.norm1 = nn.InstanceNorm3d(cout); self.film = FiLM(z_dim, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1); self.norm2 = nn.InstanceNorm3d(cout); self.act = nn.LeakyReLU(0.2, True)
        self.proj = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x, z):
        r = self.proj(x); x = self.act(self.norm1(self.conv1(x))); x = self.norm2(self.conv2(self.film(x, z))); return self.act(x + r)


class Upsample3D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.conv = nn.Conv3d(cin, cout, 3, padding=1); self.norm = nn.InstanceNorm3d(cout); self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.act(self.norm(self.conv(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))))


class FiLMDecoder(nn.Module):
    """field from the reference features (levels 3 -> 1, U-Net-like skips) conditioned on the token grid: spatial FiLM at the four skip sites, the
    token mean through FiLM in every residual block. light=True (the paper's setting): field predicted at 48^3 and trilinearly upsampled to 96^3."""
    def __init__(self, z_dim=128, skip_ch=LEVEL_CHANNELS, dec_ch=(256, 128, 96, 48), light=True):
        super().__init__(); c1, c2, c3 = skip_ch; d1, d2, d3, d4 = dec_ch; self.light = light
        self.film0, self.film1, self.film2, self.film3 = SpatialFiLM(z_dim, c3), SpatialFiLM(z_dim, d1 + c2), SpatialFiLM(z_dim, d2 + c1), SpatialFiLM(z_dim, d3)
        self.blk0, self.blk1, self.blk2, self.blk3 = ResBlock3D(c3, d1, z_dim), ResBlock3D(d1 + c2, d2, z_dim), ResBlock3D(d2 + c1, d3, z_dim), ResBlock3D(d3, d4, z_dim)
        self.up0, self.up1, self.up2 = Upsample3D(d1, d1), Upsample3D(d2, d2), Upsample3D(d3, d3)
        if not light: self.up3 = Upsample3D(d4, d4)
        self.final = ResBlock3D(d4, d4, z_dim); self.out = nn.Conv3d(d4, 3, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, ref, tokens):                                           # ref {level: (B, C, ...)} (frame-0 features), tokens (B, N, d) -> (B, 3, 96^3)
        n = round(tokens.shape[1] ** (1 / 3)); z = tokens.mean(1); grid = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, n, n, n)
        x = self.up0(self.blk0(self.film0(ref[3], grid), z))
        x = self.up1(self.blk1(self.film1(torch.cat([x, ref[2]], 1), grid), z))
        x = self.up2(self.blk2(self.film2(torch.cat([x, ref[1]], 1), grid), z))
        x = self.blk3(self.film3(x, grid), z)
        if self.light: return 2 * F.interpolate(self.out(self.final(x, z)), scale_factor=2, mode="trilinear", align_corners=False)   # 48^3 voxels -> 96^3 voxels
        return self.out(self.final(self.up3(x), z))


class SpatialTransformer(nn.Module):
    """warp src (B, C, D, H, W) by u (B, 3, D, H, W) in voxels: out(x) = src(x + u(x)), trilinear"""
    def __init__(self, size=(96, 96, 96)):
        super().__init__()
        self.register_buffer("grid", torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s) for s in size], indexing="ij"), -1)[None])

    def forward(self, src, u):
        D, H, W = src.shape[-3:]; norm = torch.tensor([2 / (D - 1), 2 / (H - 1), 2 / (W - 1)], device=u.device, dtype=u.dtype)
        g = self.grid.expand(src.shape[0], -1, -1, -1, -1) + u.permute(0, 2, 3, 4, 1) * norm
        return F.grid_sample(src, g[..., [2, 1, 0]], align_corners=True, mode="bilinear", padding_mode="zeros")


def build_teacher(cfg):
    return MotionEncoder(cfg.get("token_dim", 128), cfg.get("token_grid", 6)), FiLMDecoder(cfg.get("token_dim", 128), light=cfg.get("light_decoder", True))


def load_teacher(path, device="cuda"):
    """(encoder, decoder, config) from a teacher checkpoint (also reads checkpoints of the original research code, keys 'motionvit' / 'decoder')"""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("encoder", ck.get("motionvit")); sd = {k: v for k, v in sd.items() if k.startswith(("feat_enc.", "frame_encoder."))}
    n_tok, d = sd["frame_encoder.spatial_pos.weight"].shape                  # architecture read from the weights (older checkpoints do not record it)
    cfg = dict(ck.get("config", {}), token_dim=d, token_grid=round(n_tok ** (1 / 3)), light_decoder=not any(k.startswith("up3.") for k in ck["decoder"]))
    enc, dec = build_teacher(cfg)
    enc.load_state_dict(sd); dec.load_state_dict(ck["decoder"])
    return enc.to(device).eval(), dec.to(device).eval(), cfg


# ---------------------------------------------------------------- student
class PlaneEncoder2D(nn.Module):
    """96^2 plane -> 6 x 6 x 256 (four stride-2 3x3 convolutions, instance normalisation, LeakyReLU 0.2)"""
    def __init__(self, feat_dim=256):
        super().__init__()
        L = lambda i, o: [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.InstanceNorm2d(o), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*L(1, 32), *L(32, 64), *L(64, 128), *L(128, feat_dim))

    def forward(self, x):
        return self.net(x)


class Student(nn.Module):
    """the three live planes S_{t-2}, S_{t-1}, S_t and the reference plane -> shared 2D CNN -> 36 plane tokens (window features minus the reference
    feature, concatenated with it); 216 grid tokens = the standardised pooled frame-0 encoder features (672 ch) of each position; one transformer over
    the 36 + 216 tokens; the 216 grid outputs regress the teacher's standardised tokens of frame t."""
    def __init__(self, d=256, layers=6, heads=8, ch=672, tok=128, n_grid=216):
        super().__init__(); self.n_grid = n_grid
        self.cnn = PlaneEncoder2D(256)
        self.plane_in = nn.Linear(4 * 256, d); self.plane_pos = nn.Parameter(torch.randn(36, d) * 0.02)
        self.grid_in = nn.Linear(ch, d); self.grid_pos = nn.Parameter(torch.randn(n_grid, d) * 0.02); self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True), layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, tok))

    def forward(self, win, ref, F0):                                          # win (B, 3, 96, 96), ref (B, 1, 96, 96), F0 (B, 216, 672) standardised
        fr = self.cnn(ref); fw = [self.cnn(win[:, i:i + 1]) - fr for i in range(3)]
        p = self.plane_in(torch.cat(fw + [fr], 1).flatten(2).transpose(1, 2)) + self.plane_pos + self.type_emb[0]
        g = self.grid_in(F0) + self.grid_pos + self.type_emb[1]
        return self.out(self.enc(torch.cat([p, g], 1))[:, 36:])               # (B, 216, 128) standardised tokens
