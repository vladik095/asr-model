import math

import torch
import torch.nn.functional as F
from torch import nn


class RelPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, T, d_model = x.shape
        device = x.device

        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device)
            * -(torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )

        pos_embedding = torch.zeros(2 * T - 1, d_model, device=device)

        pos = torch.arange(-(T - 1), T, device=device).unsqueeze(1)

        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)

        return pos_embedding


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.head_dim = int(d_model / num_heads)
        self.d_model = d_model

        self.u = nn.Parameter(torch.Tensor(num_heads, self.head_dim).unsqueeze(0).unsqueeze(2))
        self.v = nn.Parameter(torch.Tensor(num_heads, self.head_dim).unsqueeze(0).unsqueeze(2))

        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value, pos_embedding, mask=None):
        batch_size = query.shape[0]

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        pos_embedding = self.pos_proj(pos_embedding)
        
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        pos_embedding = pos_embedding.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        content_score = torch.matmul((query + self.u), key.transpose(2, 3))
        pos_score = torch.matmul(
            (query + self.v), pos_embedding.transpose(2, 3)
        )  # B H T 2T-1

        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, dim=-1)

        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        # batch_size, num_heads, seq_len1, seq_len2 = pos_score.shape
        B, H, T, _ = pos_score.shape

        indices = torch.arange(T, device=pos_score.device)

        relative_indices = indices[None, :] - indices[:, None]
        relative_indices = relative_indices + T - 1

        pos_score = pos_score[:, :, torch.arange(T)[:, None], relative_indices]

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.pos_encoding = RelPositionalEncoding()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        pos_embedding = self.pos_encoding(x)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        x = self.layer_norm(x)

        out = self.attention(x, x, x, pos_embedding=pos_embedding, mask=mask)

        return out
