from torch import nn, einsum
from einops import rearrange
import math, torch
from ..utils.ch_discover_loss import DynamicalContrastiveLoss


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x, attn_mask=None):
        # projections
        h = self.heads
        q = self.to_q(x).contiguous()
        k = self.to_k(x).contiguous()
        v = self.to_v(x).contiguous()
        scale = 1.0 / self.d_k

        # shape to multi-head
        q = rearrange(q, 'b n (h d) -> b h n d', h=h).contiguous()
        k = rearrange(k, 'b n (h d) -> b h n d', h=h).contiguous()
        v = rearrange(v, 'b n (h d) -> b h n d', h=h).contiguous()

        dynamical_contrastive_loss = None

        # scores
        scores = einsum('b h i d, b h j d -> b h i j', q, k).contiguous()

        # norm matrix for DCL
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm).contiguous()

        # mask (safe, no in-place)
        if attn_mask is not None:
            am = attn_mask.to(scores.dtype).unsqueeze(1).contiguous()  # [B,1,N,N]
            large_neg = torch.tensor(-1e9, dtype=scores.dtype, device=scores.device)
            masked_scores = (scores * am + (1.0 - am) * large_neg).contiguous()

            # avoid rows fully -inf → open self (0) to prevent NaN in softmax
            B, H, N, _ = masked_scores.shape
            row_all_blocked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)
            if row_all_blocked.any():
                eye = torch.eye(N, device=masked_scores.device, dtype=masked_scores.dtype).view(1, 1, N, N)
                masked_scores = torch.where(row_all_blocked, eye * 0.0, masked_scores).contiguous()

            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores

        # attention
        attn = self.attend(masked_scores * scale).contiguous()
        out = einsum('b h i j, b h j d -> b h i d', attn, v).contiguous()
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()

        return self.to_out(out).contiguous(), attn, dynamical_contrastive_loss


class c_Transformer(nn.Module):  ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, regular_lambda=regular_lambda,
                                    temperature=temperature)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        total_loss = None
        last_attn = None
        for attn, ff in self.layers:
            x_n, last_attn, dcloss = attn(x, attn_mask=attn_mask)
            total_loss = dcloss if total_loss is None else (total_loss + dcloss)
            x = (x_n + x).contiguous()
            x = (ff(x) + x).contiguous()
        if total_loss is None:
            # fallback when DCL returns None
            total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        dcloss = total_loss / len(self.layers)
        return x.contiguous(), last_attn, dcloss


class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                         regular_lambda=regular_lambda, temperature=temperature)

        self.mlp_head = nn.Linear(dim, d_model)  # horizon)

    def forward(self, x, attn_mask=None):
        # [B*patch, C, patch_dim] -> [B*patch, C, dim]
        x = self.to_patch_embedding(x).contiguous()
        x, attn, dcloss = self.transformer(x.contiguous(), attn_mask)
        x = self.dropout(x).contiguous()
        x = self.mlp_head(x).contiguous()   # [B*patch, C, d_model]
        # squeeze() 제거: dim 지정 없는 squeeze는 뷰/버전 충돌 원인
        return x, dcloss  # ,attn
