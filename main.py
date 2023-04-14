import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, device, dropout=0.1):
        """
        :param embedding_dim: input dimension
        :param n_heads: number of heads
        :param dropout_ratio:
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_model, d_k * n_heads)
        self.fc_k = nn.Linear(d_model, d_k * n_heads)
        self.fc_v = nn.Linear(d_model, d_v * n_heads)
        self.fc_out = nn.Linear(d_v * n_heads, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        """
        :param key, query, value: [batch_size X sequence_length X embedding_dim] (e.g., 32 X 10 X 512)
        :param mask: mask for decoder
        """
        d_k = self.d_k
        d_v = self.d_v
        n_heads = self.n_heads

        batch_size = query.size(dim=0)
        len_q = query.size(1)
        len_k = key.size(1)
        len_v = value.size(1)

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, len_q, n_heads, d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, len_k, n_heads, d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, len_v, n_heads, d_v).permute(0, 2, 1, 3)

        scale_dot_prod = torch.matmul(Q, K.transpose(3, 2)) / self.scale
        # scale_dot_prod: [batch, n_heads, len_q, len_k]

        if mask is not None:
            scale_dot_prod = scale_dot_prod.masked_fill(mask==0, -1e9)

        attention = self.dropout(F.softmax(scale_dot_prod, dim=-1))
        # attention: [batch, n_heads, len_q, len_k]
        x = torch.matmul(attention, V)
        # output: [batch, n_heads, len_q, n_heads*d_v]

        x = x.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.fc_out(x)

        return output, attention