"""
TPP-LLM Layers
"""
import math
from typing import Union

import torch
from torch import Tensor
from torch import nn

class LearnableTemporalEncoding(nn.Module):
    """
    TPE (sin/cos) の「スケール(周波数)」を学習可能にした改良版
    (Inplace operation エラーを修正済み)
    """

    def __init__(self, embedding_dim: int, dtype=torch.float32, device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        i = torch.arange(0, embedding_dim, 1, dtype=self.dtype, device=self.device)
        initial_div_term = (2 * (i // 2).to(self.dtype) * -(math.log(10000.0) / embedding_dim)).exp()
        
        self.div_term = nn.Parameter(initial_div_term)

    def forward(self, event_values: Tensor) -> Tensor:
        """
        :param event_values: イベントの連続値 (..., 1)
        :return: エンコードされたベクトル (..., embedding_dim)
        """
        
        # 1. 角度を計算 (ここは変更なし)
        # (event_values * self.div_term) の結果を "angles" という変数名に変更
        angles = (event_values * self.div_term).to(self.dtype).to(self.device)
        
        # 2. ★★★ 修正点 ★★★
        #    "angles" と同じ形状の「新しい空のテンソル」を "result" として作成
        result = torch.empty_like(angles)
    
        # 3. 「angles」から計算した結果を、「result」に書き込む
        #    (これで "angles" 自体は上書きされず、勾配計算が安全になる)
        result[..., 0::2] = torch.sin(angles[..., 0::2])
        result[..., 1::2] = torch.cos(angles[..., 1::2])
        
        return result
    
class TimePositionalEncoding(nn.Module):
    """
    Temporal Positional Encoding from THP
    """

    def __init__(self, embedding_dim: int, dtype=torch.float32, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the temporal encoding

        :param embedding_dim: embedding dimension
        :param dtype: data type for the output
        :param device: device
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        i = torch.arange(0, embedding_dim, 1, dtype=self.dtype, device=self.device)
        div_term = (2 * (i // 2).to(self.dtype) * -(math.log(10000.0) / embedding_dim)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, event_times: Tensor) -> Tensor:
        """
        Compute time positional encoding defined in the THP model

        :param event_times: event times, (seq_len, 1)
        :return: temporal encoding vector, (seq_len, embedding_dim)
        """
        result = (event_times * self.div_term).to(self.dtype).to(self.device)
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result
    
    


class TimeShiftedPositionalEncoding(nn.Module):
    """
    Time-Shifted Positional Encoding from SAHP
    """

    def __init__(
        self, embedding_dim: int, max_len: int = 5000, dtype=torch.float32, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the temporal encoding

        :param embedding_dim: embedding dimension
        :param max_len: maximum sequence length
        :param dtype: data type for the output
        :param device: device
        """
        super().__init__()
        self.dtype = dtype
        self.device = device

        position = torch.arange(0, max_len, dtype=self.dtype, device=self.device).unsqueeze(1)  # (max_len, 1)
        div_term = (
            torch.arange(0, embedding_dim, 2, dtype=self.dtype, device=self.device)
            * -(math.log(10000.0) / embedding_dim)
        ).exp()  # (model_dim // 2, )
        self.layer_time_delta = nn.Linear(1, embedding_dim // 2, bias=False, dtype=self.dtype, device=self.device)

        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

    def forward(self, event_times: Tensor, event_time_deltas: Tensor) -> Tensor:
        """
        Compute time-shifted positional encoding defined in the SAHP model

        :param event_times: event times, (seq_len, 1)
        :param event_time_deltas: event time deltas, (seq_len, 1)
        :return: temporal encoding vector, (seq_len, embedding_dim)
        """
        phi = self.layer_time_delta(event_time_deltas)
        length = event_times.size(0)
        arc = (self.position[:length] * self.div_term).to(self.dtype).to(self.device)
        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe

import torch.nn as nn

class LatLonMLPEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dtype=torch.float32, device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embedding_dim,dtype=dtype, device=device),   # 入力は2次元（lat, lon）
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim,dtype=dtype, device=device)
        )
    def forward(self, latlon):
        return self.mlp(latlon)  # shape: [B, embedding_dim]
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericEmbedding(nn.Module):
    def __init__(self, out_dim, n_exp_bins=65, exp_offset=32, rff_dim=64, rff_freq_scale=10.0):
        """
        out_dim: 最終埋め込み次元
        n_exp_bins: exponent を割り当てるビン数（例: -32..+32 -> 65）
        exp_offset: exponent に加えるオフセット（ビンインデックス化のため）
        rff_dim: mantissa を投影するRFFの次元（sin/cosなので実質2*rff_dim）
        """
        super().__init__()
        self.out_dim = out_dim
        self.n_exp_bins = n_exp_bins
        self.exp_offset = exp_offset
        self.exp_emb_dim = min(32, out_dim//2)
        self.exp_embedding = nn.Embedding(n_exp_bins, self.exp_emb_dim)
        # RFF
        self.rff_dim = rff_dim
        self.register_parameter("rff_W", nn.Parameter(torch.randn(rff_dim, 1) * rff_freq_scale))
        self.register_parameter("rff_b", nn.Parameter(torch.rand(rff_dim) * 2 * torch.pi))
        self.mantissa_proj = nn.Sequential(
            nn.Linear(2*rff_dim, (out_dim - self.exp_emb_dim)),
            nn.GELU(),
            nn.Linear((out_dim - self.exp_emb_dim), (out_dim - self.exp_emb_dim))
        )
        # scale param for monotonic loss
        self.scale_for_mono = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        x: float tensor shape [B, L]
        returns: embedding [B, L, out_dim]
        """
        B, L = x.shape
        # handle zero -> frexp undefined sign? torch.frexp handles zeros.
        mantissa, exponent = torch.frexp(x)   # mantissa in [-1,1), exponent int
        # map exponent to bins
        exp_idx = (exponent + self.exp_offset).long().clamp(0, self.n_exp_bins - 1)  # [B,L]
        exp_emb = self.exp_embedding(exp_idx)  # [B,L,exp_emb_dim]

        # mantissa -> RFF
        # make mantissa shape [B*L, 1]
        m = mantissa.reshape(-1, 1)  # values in (-1,1)
        # rff: sin(W * m + b), W: [rff_dim,1], b: [rff_dim]
        proj = torch.sin(F.linear(m, self.rff_W, self.rff_b))  # [B*L, rff_dim]
        # also use cos to double features (optional)
        proj_cos = torch.cos(F.linear(m, self.rff_W, self.rff_b))
        proj_cat = torch.cat([proj, proj_cos], dim=-1)  # [B*L, 2*rff_dim]
        mantissa_emb = self.mantissa_proj(proj_cat)    # [B*L, out_dim - exp_emb_dim]
        mantissa_emb = mantissa_emb.view(B, L, -1)

        # concat
        out = torch.cat([exp_emb, mantissa_emb], dim=-1)  # [B,L,out_dim]
        return out

    def monotonic_loss(self, x, embeddings, n_pairs=256, reduction='mean'):
        """
        x: [B,L] floats
        embeddings: [B,L,D] numeric embeddings
        samples random pairs across batch/time and compute monotonic loss.
        """
        B, L = x.shape
        N_total = B*L
        flat_x = x.reshape(-1)
        flat_e = embeddings.reshape(N_total, -1)
        # sample indices
        idx = torch.randint(0, N_total, (n_pairs*2,), device=x.device)
        i = idx[:n_pairs]
        j = idx[n_pairs:]
        xi = flat_x[i]
        xj = flat_x[j]
        ei = flat_e[i]
        ej = flat_e[j]
        emb_dist = torch.norm(ei - ej, dim=-1)  # [n_pairs]
        numeric_diff = torch.abs(xi - xj)
        # optionally stabilise numeric diff
        # numeric_diff = torch.log1p(numeric_diff)
        scaled = self.scale_for_mono * numeric_diff
        loss = F.mse_loss(emb_dist, scaled, reduction=reduction)
        return loss