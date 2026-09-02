from importlib_metadata import version
# print("torch version: ", version("torch")) 2.8.0

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionKVCache(nn.Module):
    """MultiHead Attention with KV Cache"""
    def __init__(self, dim=512, n_heads=8, use_kv=True):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_kv = use_kv

        # Wq, Wk, Wv, Wo Matrix
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # story historical tokens
        self.history_seq = []
    
    def forward(self, q, k, v, past_key=None, past_value=None, mask=None):
        # Prefill phase: seq_len > 1
        # Decoding phase: seq_len = 1
        batch_size, seq_len, _ = q.shape

        if not self.use_kv:
            self.history_seq.append(q)

        # Q, K, V mapping
        q = self.q_proj(q)
        if self.use_kv:
            k = self.k_proj(k)
            v = self.v_proj(v)

        # divide multihead, shape = [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_kv:
            k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_kv:
            # KV Cache concatenates past_key, past_value in seq_len dimension
            # history: batch, n_heads, seq_len, head_dim
            # new: batch, n_heads, 1, head_dim
            # after concatenation: batch, n_heads, seq_len + 1, head_dim
            if past_key is not None:
                k = torch.cat([past_key, k], dim=2)
            if past_value is not None:
                v = torch.cat([past_value, v], dim=2)
        else:
            # standard MHA
            history_seq = torch.cat(self.history_seq, dim=1)
            k = self.k_proj(history_seq)
            v = self.v_proj(history_seq)
            k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # store k, v for next prediction 
        past_key_values = (k, v)

        # MHA scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # casual mask
        if mask is not None:
            # mask is a bool matrix, in which True represents contents will be masked 
            attn_scores = attn_scores.masked_fill(mask, 1e-9)

        # softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # batch, n_heads, seq_len, head_dim
        attn_output = torch.matmul(attn_weights, v)

        # concatenate MHA
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)

        # output mapping
        output = self.o_proj(attn_output)
        return output, past_key_values

def calculate_kv_cache_size(past_key_values):
    # calculate past_key_values in memory (MB)

    total_size = 0
    if past_key_values is None:
        return total_size
    
    # past_key_values: k, v
    for tensor in past_key_values:
        # numel() returns # of elements, element_size() returns size of every element
        total_size += tensor.numel() * tensor.element_size()
    
    return total_size / (1024 * 1024)