import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, feats=32, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(feats, feats // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feats // reduction, feats, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = self.sigmoid(x)
        return y * res

class CBAM(nn.Module):
    def __init__(self, feats=32, kernel_size=7, reduction=16):
        super(CBAM,self).__init__()
        self.ca=ChannelAttention(feats)
        self.sa=SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        x = self.sa(self.ca(x))
        return x
def _batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    values  : [B, L, C]
    indices : [B, K]
    returns : [B, K, C]
    """
    C = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, C))
class VSA(nn.Module):
    """
    channels   : number of feature channels C
    chunk_size : tokens per attention chunk (see memory table in module doc)
    n_hashes   : number of independent LSH rounds
    reduction  : key/query channel ratio  (C_r = C // reduction)
    res_scale  : residual scaling factor
    """

    def __init__(
        self,
        channels:   int,
        chunk_size: int   = 64,
        n_hashes:   int   = 4,
        reduction:  int   = 4,
        res_scale:  float = 1.0,
    ):
        super().__init__()
        assert channels % reduction == 0, \
            f"channels ({channels}) must be divisible by reduction ({reduction})"

        self.chunk_size = chunk_size
        self.n_hashes   = n_hashes
        self.reduction  = reduction
        self.res_scale  = res_scale

        C_r = channels // reduction
        self.conv_match = nn.Conv2d(
            channels, C_r, kernel_size=3, padding=1, bias=True
        )
        self.conv_assembly = nn.Conv2d(
            channels, channels, kernel_size=1, bias=True
        )

    def _lsh(self, hash_buckets: int, x: torch.Tensor) -> torch.Tensor:
        """
        x           : [B, L, C_r]    L = UV*H*W
        hash_buckets: even integer
        returns     : [B, n_hashes * L]  – hash code per token per round
        """
        B     = x.shape[0]
        device = x.device
        rot_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)
        rand_rot  = (torch.randn(rot_shape, dtype=x.dtype, device=device)
                     .expand(B, -1, -1, -1))
        rotated = torch.einsum('btf,bfhi->bhti', x, rand_rot)
        rotated = torch.cat([rotated, -rotated], dim=-1)        # [B, h, L, B]
        codes  = torch.argmax(rotated, dim=-1)                  # [B, h, L]
        offset = torch.arange(self.n_hashes, device=device)
        offset = (offset * hash_buckets).reshape(1, -1, 1)
        codes  = (codes + offset).reshape(B, -1)                # [B, h*L]
        return codes
    @staticmethod
    def _add_adjacent(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_hashes, num_chunks, chunk_size, C]
        Concatenates the previous and next chunk along dim=3 (chunk_size axis)
        so each position can attend to its own chunk ± one neighbour.
        Returns: [B, n_hashes, num_chunks, 3*chunk_size, C]
        """
        back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        fwd  = torch.cat([x[:, :,  1:, ...], x[:, :, :1,  ...]], dim=2)
        return torch.cat([x, back, fwd], dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, U, V, C, H, W]
        returns: [B, U, V, C, H, W]
        """
        B, U, V, C, H, W = x.shape
        UV  = U * V
        L   = UV * H * W          # full token-sequence length
        C_r = C // self.reduction  # key/query channels

        # ── 1. Per-view feature extraction ───────────────────────────────
        x_2d = x.reshape(B * UV, C, H, W)

        # key/query features  [B*UV, C_r, H, W]
        kq = self.conv_match(x_2d)
        # value features      [B*UV, C,   H, W]
        val = self.conv_assembly(x_2d)

        # ── 2. Flatten all views → single token sequence ─────────────────
        # x_embed : [B, L, C_r]    y_embed : [B, L, C]
        x_embed = (kq
                   .permute(0, 2, 3, 1)              # [B*UV, H, W, C_r]
                   .reshape(B, UV, H, W, C_r)
                   .reshape(B, L, C_r))

        y_embed = (val
                   .permute(0, 2, 3, 1)              # [B*UV, H, W, C]
                   .reshape(B, UV, H, W, C)
                   .reshape(B, L, C))

        # ── 3. LSH hashing & sort ─────────────────────────────────────────
        hash_buckets = min(
            L // self.chunk_size + (L // self.chunk_size) % 2,
            128
        )
        hash_codes = self._lsh(hash_buckets, x_embed).detach()   # [B, n_hashes*L]

        _, indices   = hash_codes.sort(dim=-1)                    # [B, n_hashes*L]
        _, undo_sort = indices.sort(dim=-1)
        mod_idx      = indices % L

        # Gather tokens in bucket order
        x_sorted = _batched_index_select(x_embed, mod_idx)       # [B, n_hashes*L, C_r]
        y_sorted = _batched_index_select(y_embed, mod_idx)        # [B, n_hashes*L, C]

        # ── 4. Pad, chunk, and compute attention ──────────────────────────
        padding = (self.chunk_size - L % self.chunk_size) % self.chunk_size

        # [B, n_hashes, L, C_r/C]
        x_bkt = x_sorted.reshape(B, self.n_hashes, -1, C_r)
        y_bkt = y_sorted.reshape(B, self.n_hashes, -1, C)

        if padding:
            x_bkt = torch.cat([x_bkt, x_bkt[:, :, -padding:, :]], dim=2)
            y_bkt = torch.cat([y_bkt, y_bkt[:, :, -padding:, :]], dim=2)

        # [B, n_hashes, num_chunks, chunk_size, C_r/C]
        x_bkt = x_bkt.reshape(B, self.n_hashes, -1, self.chunk_size, C_r)
        y_bkt = y_bkt.reshape(B, self.n_hashes, -1, self.chunk_size, C)

        # Normalised keys (adjacent-extended); unnormalised queries
        x_norm = F.normalize(x_bkt, p=2.0, dim=-1, eps=5e-5)
        x_keys = self._add_adjacent(x_norm)  # [B, h, nc, 3*cs, C_r]
        y_vals = self._add_adjacent(y_bkt)   # [B, h, nc, 3*cs, C]

        # Attention scores  [B, n_hashes, num_chunks, chunk_size, 3*chunk_size]
        # Queries: unnormalised x_bkt; Keys: normalised x_keys
        raw_score = torch.einsum('bhkic,bhkjc->bhkij', x_bkt, x_keys)

        # Stable softmax via log-sum-exp
        log_z     = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score     = torch.exp(raw_score - log_z)   # [B, h, nc, cs, 3*cs]

        # Bucket log-normaliser for multi-round weighting  [B, h, nc*cs]
        bkt_score = log_z.squeeze(-1).reshape(B, self.n_hashes, -1)

        # Value aggregation  [B, n_hashes, num_chunks, chunk_size, C]
        ret = torch.einsum('bhkij,bhkjc->bhkic', score, y_vals)
        ret = ret.reshape(B, self.n_hashes, -1, C)                # [B, h, nc*cs, C]

        # ── 5. Remove padding, recover original order ─────────────────────
        if padding:
            ret       = ret[:, :, :-padding, :].clone()
            bkt_score = bkt_score[:, :, :-padding].clone()

        ret       = ret.reshape(B, -1, C)           # [B, n_hashes*L, C]
        bkt_score = bkt_score.reshape(B, -1)        # [B, n_hashes*L]

        ret       = _batched_index_select(ret, undo_sort)
        bkt_score = bkt_score.gather(1, undo_sort)

        # ── 6. Multi-round weighted aggregation ───────────────────────────
        ret       = ret.reshape(B, self.n_hashes, L, C)
        bkt_score = bkt_score.reshape(B, self.n_hashes, L, 1)
        probs     = F.softmax(bkt_score, dim=1)          # weight across hash rounds
        ret       = (ret * probs).sum(dim=1)             # [B, L, C]

        # ── 7. Reshape back to light-field format + residual ──────────────
        ret = (ret
               .reshape(B, UV, H, W, C)
               .permute(0, 1, 4, 2, 3)               # [B, UV, C, H, W]
               .reshape(B, U, V, C, H, W))

        return ret * self.res_scale + x
class VSANet(nn.Module):
    def __init__(self, opt):
        super(VSANet, self).__init__()
        self.n_groups = opt.n_groups
        self.n_blocks = opt.n_blocks
        self.n_channels  = opt.n_channels
        self.conv_first = nn.Conv2d(3, self.n_channels,  kernel_size=3, stride=1, dilation=1, padding=1, bias=True)
        Groups = [
            SEAR(self.n_blocks, self.n_channels) \
            for _ in range(self.n_groups)]
        self.Group = nn.Sequential(*Groups)
        self.conv_last = nn.Conv2d(self.n_channels, 3,  kernel_size=3, stride=1, dilation=1, padding=1, bias=True)
    def forward(self, x):
        b,u,v,c,h,w = x.shape
        out = x.reshape(b*u*v,c,h,w)
        out = self.conv_first(out)
        out = out.reshape(b,u,v,self.n_channels,h,w)
        out = self.Group(out)
        out = out.reshape(b*u*v,self.n_channels,h,w)
        out = self.conv_last(out)
        out = out.reshape(b,u,v,c,h,w)
        out += x
        return out
class SEAR(nn.Module):
    def __init__(self ,n_blocks, n_channels):
        super(SEAR, self).__init__()
        self.fea_block = make_layer(SFE, n_channels, n_blocks)
        self.VSA = VSA(channels=n_channels,chunk_size = 64,
                    n_hashes=6,reduction=2,res_scale=1.0,)        
        self.FR = FR(n_channels)
        
    def forward(self, x):
        out = self.fea_block(x)
        out = self.VSA(out)
        out = self.FR(out)
        out += x
        return out
class SFE(nn.Module):
    def __init__(self, n_channels):
        super(SFE, self).__init__()
        self.SSAB = SSAB(n_channels)
        self.EPIB = EPIB(n_channels)
    def forward(self, x):
        out_SSAB = self.SSAB(x)
        out_EPIB = self.EPIB(x)
        out = out_SSAB + out_EPIB + x
        return out
class SSAB(nn.Module):
    def __init__(self, n_channels):
        super(SSAB, self).__init__()
        self.conv_ss = torch.nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_as = torch.nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.n_channels = n_channels
    def forward(self, x):
        b,u,v,c,h,w = x.shape
        out= self.act(self.conv_ss(x.reshape(b*u*v,c,h,w))) #[64]
        out= self.act(self.conv_as(out.reshape(b,u,v,self.n_channels,h,w).permute(0,4,5,3,1,2).reshape(b*h*w,self.n_channels,u,v))) #[64]
        out = out.reshape(b,h,w,self.n_channels,u,v).permute(0,4,5,3,1,2)
        return out
class EPIB(nn.Module):
    def __init__(self, n_channels):
        super(EPIB, self).__init__()
        self.conv_ehs =torch.nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_evs =torch.nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.n_feat = n_channels
    def forward(self, x):
        b,u,v,c,h,w = x.shape
        out = x.permute(0, 2, 5, 3, 1, 4).contiguous()
        out = out.reshape(b * v * w, c, u, h)
        out = self.act(self.conv_evs(out))
        out = out.reshape(b, v, w, c, u, h).permute(0,4,5,3,1,2)
        out = out.reshape(b * u * h, c, v, w)  
        out = self.act(self.conv_ehs(out))
        out = out.reshape(b, u, h, c, v, w).permute(0, 1, 4, 3, 2, 5).contiguous() 
        return out
class FR(nn.Module):
    def __init__(self,in_channels):
        super(FR, self).__init__()
        self.att1 = CBAM(feats=in_channels,kernel_size=7)
        self.att2 = CBAM(feats=in_channels,kernel_size=3)
        self.conv = nn.Conv2d(in_channels*4, in_channels,  kernel_size=3, stride=1, dilation=1, padding=1, bias=True)
    def forward(self,x):
        b,u,v,c,h,w = x.shape
        ss_input = x.reshape(b*u*v,c,h,w)
        ss_out = self.att1(ss_input)  #[b*u*v,c,h,w]
        as_input = x.permute(0,4,5,3,1,2).reshape(b*h*w,c,u,v)
        as_out = self.att2(as_input)
        as_out = as_out.reshape(b, h, w, c, u, v).permute(0, 4, 5, 3, 1, 2).reshape(b*u*v,c,h,w) #[b*u*v,c,h,w]
        ehs_input = x.permute(0,2,5,3,1,4).reshape(b*v*w,c,u,h)
        ehs_ouput = self.att2(ehs_input)
        ehs_ouput = ehs_ouput.reshape(b, v, w, c, u, h).permute(0, 4, 1, 3, 5, 2).reshape(b*u*v,c,h,w) #[b*u*v,c,h,w]
        evs_input = x.permute(0,1,4,3,2,5).reshape(b*u*h,c,v,w)
        evs_ouput = self.att2(evs_input)
        evs_ouput = evs_ouput.reshape(b, u, h, c, v, w).permute(0, 1, 4, 3, 2, 5).reshape(b*u*v,c,h,w) #[b*u*v,c,h,w]
        merged = torch.cat((ss_out, as_out, ehs_ouput, evs_ouput), 1)
        out = self.conv(merged)
        out = out.reshape(b, u, v, c, h, w).contiguous() #[b,c,u,v,h,w]
        return out

def make_layer(block, nf, n_layers ):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf))
    return nn.Sequential(*layers)	
   
        