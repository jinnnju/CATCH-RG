'''
* @author: EmpyreanMoon
* @create: 2024-08-26 10:28
* @description: The structure of CATCH (safe contiguous ver. + RevIN clone)
'''

from ts_benchmark.baselines.catch.layers.RevIN import RevIN
from ts_benchmark.baselines.catch.layers.cross_channel_Transformer import Trans_C
# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.catch.layers.channel_mask import channel_mask_generator


class CATCHModel(nn.Module):
    def __init__(self, configs,
                 **kwargs):
        super(CATCHModel, self).__init__()

        self.revin_layer = RevIN(configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last)
        # Patching
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = self.seq_len
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)
        # print("depth=",cf_depth)
        # Backbone
        self.re_attn = True
        self.mask_generator = channel_mask_generator(input_size=configs.patch_size, n_vars=configs.c_in)
        self.frequency_transformer = Trans_C(dim=configs.cf_dim, depth=configs.e_layers, heads=configs.n_heads,
                                       mlp_dim=configs.d_ff,
                                       dim_head=configs.head_dim, dropout=configs.dropout,
                                       patch_dim=configs.patch_size * 2,
                                       horizon=self.horizon * 2, d_model=configs.d_model * 2,
                                       regular_lambda=configs.regular_lambda, temperature=configs.temperature)

        # Head
        self.head_nf_f = configs.d_model * 2 * patch_num
        self.n_vars = configs.c_in
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, configs.seq_len,
                                    head_dropout=configs.head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, configs.seq_len,
                                    head_dropout=configs.head_dropout)

        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)

        # break up R&I:
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def forward(self, z):  # z: [bs x seq_len x n_vars]
        # RevIN 앞단에서 뷰 안전화
        z = z.contiguous()
        # RevIN 내부 in-place 가능성에 대비해 결과를 clone()하여 버전 카운터 분리
        z = self.revin_layer(z, 'norm').clone()

        # [B, L, C] -> [B, C, L]
        z = z.permute(0, 2, 1).contiguous()

        # FFT (복소)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        # do patching (unfold는 as_strided 뷰 → view 끊기)
        z1 = z1.unfold(dimension=-1, size=self.patch_size,
                       step=self.patch_stride).contiguous().clone()  # [bs, nvars, patch_num, patch_size]
        z2 = z2.unfold(dimension=-1, size=self.patch_size,
                       step=self.patch_stride).contiguous().clone()

        # for channel-wise_1 (permute 후 contiguous)
        z1 = z1.permute(0, 2, 1, 3).contiguous()
        z2 = z2.permute(0, 2, 1, 3).contiguous()

        # model shape
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        # patch_size = z1.shape[3]

        # [bs * patch_num, nvars, patch_size]
        z1 = z1.reshape(batch_size * patch_num, c_in, z1.shape[-1]).contiguous()
        z2 = z2.reshape(batch_size * patch_num, c_in, z2.shape[-1]).contiguous()
        z_cat = torch.cat((z1, z2), -1).contiguous()

        channel_mask = self.mask_generator(z_cat)  # [B*patch, C, C]

        z, dcloss = self.frequency_transformer(z_cat, channel_mask)  # [B*patch, C, d_model*2]

        # 분리
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        # [B, patch_num, C, d_model*2]
        z1 = z1.reshape(batch_size, patch_num, c_in, z1.shape[-1]).contiguous()
        z2 = z2.reshape(batch_size, patch_num, c_in, z2.shape[-1]).contiguous()

        # [B, C, patch_num, horizon]
        z1 = z1.permute(0, 2, 1, 3).contiguous()
        z2 = z2.permute(0, 2, 1, 3).contiguous()

        # Head
        z1 = self.head_f1(z1)  # [B, C, L]
        z2 = self.head_f2(z2)  # [B, C, L]

        complex_z = torch.complex(z1, z2)

        z = torch.fft.ifft(complex_z)  # [B, C, L] 복소
        zr = z.real.contiguous()
        zi = z.imag.contiguous()
        z = self.ircom(torch.cat((zr, zi), -1).contiguous())

        # denorm
        z = z.permute(0, 2, 1).contiguous()
        # RevIN 후단도 clone()으로 버전 분리
        z = self.revin_layer(z, 'denorm').clone()

        return z, complex_z.permute(0, 2, 1).contiguous(), dcloss


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, seq_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :]).contiguous()  # [bs x d_model * patch_num]
                z = self.linears1[i](z)                           # [bs x seq_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1).contiguous()  # [bs x nvars x seq_len]
        else:
            x = self.flatten(x).contiguous()            # [bs, nvars, nf]
            # residual은 in-place 아님: 안전
            h = F.relu(self.linear1(x))
            x = h + x
            h = F.relu(self.linear2(x))
            x = h + x
            h = F.relu(self.linear3(x))
            x = h + x
            x = self.linear4(x)

        return x
