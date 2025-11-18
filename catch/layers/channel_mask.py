'''
* @author: EmpyreanMoon
* @create: 2024-09-02 17:32
* @description: (입출력 동일) Robust channel_mask_generator
                – 자동 신뢰도 추정 + soft gating for missing/noisy channels
                – all-channel down 시 identity 고정
'''
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import gumbel_softmax
import torch.nn.functional as F


class channel_mask_generator(nn.Module):
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()
        # === 원래 구조 유지 ===
        self.generator = nn.Sequential(
            nn.Linear(input_size * 2, n_vars, bias=False),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.generator[0].weight.zero_()

        self.n_vars = n_vars

        # === 추가: 채널 신뢰도 추정 모듈 (작고 가벼움) ===
        self.reliability_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.eps = 1e-6

    def forward(self, x):  # x: [(bs x patch_num) x n_vars x patch_size]
        Bp, C, P = x.shape

        # 뷰/NaN 안전화
        x_safe = torch.nan_to_num(x.contiguous(), nan=0.0, posinf=0.0, neginf=0.0)

        # --- (0) 전사 다운(모든 채널·시간 거의 0) 감지 ---
        down_flag = (x_safe.abs().sum(dim=(1, 2)) < 1e-12)  # [Bp] bool

        # (1) 기본 분포 예측
        distribution_matrix = self.generator(x_safe).clamp(self.eps, 1 - self.eps)  # [Bp, C, C]

        # (2) 간단한 채널 신뢰도 계산 → soft gating
        var = x_safe.var(dim=-1, unbiased=False).clamp_min(1e-8)
        logvar = torch.log(var)
        absmean = x_safe.abs().mean(dim=-1)
        diff = (x_safe[..., 1:] - x_safe[..., :-1]).abs().max(dim=-1).values
        zero_ratio = (x_safe.abs() < 1e-6).float().mean(dim=-1)

        stats = torch.stack([logvar, absmean, diff, zero_ratio], dim=-1).contiguous()  # [Bp, C, 4]
        r = self.reliability_head(stats).squeeze(-1).clamp_min(0.05)                   # [Bp, C]
        r_outer = (r.unsqueeze(-1) * r.unsqueeze(-2)).contiguous()                     # [Bp, C, C]

        distribution_matrix = (distribution_matrix * r_outer).clamp(self.eps, 1 - self.eps)

        # (3) Gumbel-Softmax 샘플링
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix).contiguous()

        # (4) 대각은 항상 1 (off-diag에만 샘플 적용)
        inverse_eye = (1 - torch.eye(self.n_vars, device=x.device)).to(resample_matrix.dtype)  # [C, C]
        diag = torch.eye(self.n_vars, device=x.device).to(resample_matrix.dtype)

        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye).contiguous()
        resample_matrix = (resample_matrix + diag).contiguous()  # [Bp, C, C]

        # (5) 전사 다운 샘플은 항등행렬로 고정 (cross-channel 완전 차단)
        if down_flag.any():
            I = diag.unsqueeze(0).expand(Bp, -1, -1)    # [Bp, C, C]
            df = down_flag.view(Bp, 1, 1)               # [Bp,1,1] bool
            resample_matrix = torch.where(df, I, resample_matrix).contiguous()

        return resample_matrix  # === 출력 동일 ===

    def _bernoulli_gumbel_rsample(self, p):
        """
        Out-of-place straight-through Bernoulli sampler compatible with autograd.
        입력/출력 shape 동일: p ∈ [B, C, D], 반환 ∈ {0,1}^{B,C,D}
        """
        b, c, d = p.shape
        p = p.clamp(self.eps, 1 - self.eps).contiguous()

        # 2-class logits: [p, 1-p]
        # logits = log(p/(1-p))  (숫자 안정화 위해 log1p 사용)
        logit = torch.log(p) - torch.log1p(-p)                   # [B,C,D]
        # Gumbel noise
        g = -torch.log(-torch.log(torch.rand_like(logit)))       # [B,C,D]
        # temperature=1.0 가정 (원래 gumbel_softmax 기본과 동일)
        y = torch.sigmoid(logit + g).contiguous()                # “soft” Bernoulli prob, [B,C,D]

        # 2-class로 확장 (첫 채널이 1 선택 확률)
        y2 = torch.stack([y, 1.0 - y], dim=-1).contiguous()      # [B,C,D,2]

        # hard one-hot (out-of-place): in-place scatter_ 대신 one_hot 사용
        hard_idx = y2.argmax(dim=-1)                             # [B,C,D] in {0,1}
        y2_hard = F.one_hot(hard_idx, num_classes=2).to(y2.dtype)  # [B,C,D,2]

        # straight-through: forward는 hard, backward는 soft
        y2_st = y2 + (y2_hard - y2).detach()                     # [B,C,D,2]

        # 첫 채널(“1을 뽑았는지”)을 마스크로 사용
        out = y2_st[..., 0]                                      # [B,C,D]
        return out.contiguous()