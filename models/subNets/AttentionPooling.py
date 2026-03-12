# *_*coding:utf-8 *_*
"""
Attention Pooling: 用 K 个可学习 query 对序列做 cross-attention，得到 K 个 token
输入 (B, T, D) -> 输出 (B, K, D)
"""
import torch
import torch.nn as nn
import math

__all__ = ['AttentionPooling']


class AttentionPooling(nn.Module):
    def __init__(self, dim, num_queries=4, num_heads=4, dropout=0.1):
        """
        Args:
            dim: 特征维度 D
            num_queries: 输出 token 数量 K
            num_heads: 注意力头数
            dropout: dropout
        """
        super(AttentionPooling, self).__init__()
        dim = int(round(dim))
        num_queries = max(1, int(round(num_queries)))
        num_heads = min(num_heads, dim)
        self.dim = dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = math.sqrt(self.head_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.proj_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D)
            mask: optional (B, T), 1 = valid, 0 = pad
        Returns:
            out: (B, K, D)
        """
        B, T, D = x.shape
        Q = self.queries.expand(B, -1, -1)  # (B, K, D)
        # single-head style: attn = softmax(Q K^T / sqrt(d))
        Q_ = Q.view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K, d)
        K_ = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, T, d)
        V_ = K_
        scores = torch.matmul(Q_, K_.transpose(-2, -1)) / self.scale  # (B, H, K, T)
        if mask is not None:
            # mask (B, T) -> (B, 1, 1, T)
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V_)  # (B, H, K, d)
        out = out.transpose(1, 2).contiguous().view(B, self.num_queries, D)
        out = self.proj_out(out)
        return out
