import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, value):
        scale = torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return value * scale.to(value.dtype) * self.weight


def apply_interleaved_rope(value, cosine, sine, start):
    sequence_length = value.size(2)
    cosine = cosine[start : start + sequence_length].view(1, 1, sequence_length, -1)
    sine = sine[start : start + sequence_length].view(1, 1, sequence_length, -1)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, heads, capacity):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.capacity = capacity
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        positions = torch.arange(capacity, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.head_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.head_dim)
        )
        angles = positions * frequencies.unsqueeze(0)
        self.register_buffer("rope_cosine", torch.cos(angles))
        self.register_buffer("rope_sine", torch.sin(angles))

    def _heads(self, value):
        batch, sequence_length, _ = value.shape
        return value.view(batch, sequence_length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, value, past_key=None, past_value=None):
        batch, sequence_length, _ = value.shape
        previous_length = 0 if past_key is None else past_key.size(2)
        if previous_length + sequence_length > self.capacity:
            raise ValueError("decoder KV cache capacity exceeded")

        query = apply_interleaved_rope(
            self._heads(self.q_proj(value)), self.rope_cosine, self.rope_sine, previous_length
        )
        key = apply_interleaved_rope(
            self._heads(self.k_proj(value)), self.rope_cosine, self.rope_sine, previous_length
        )
        new_value = self._heads(self.v_proj(value))
        key_cache = key if past_key is None else torch.cat((past_key, key), dim=2)
        value_cache = (
            new_value if past_value is None else torch.cat((past_value, new_value), dim=2)
        )

        scores = torch.matmul(query, key_cache.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_positions = torch.arange(
            previous_length,
            previous_length + sequence_length,
            device=value.device,
        ).view(sequence_length, 1)
        key_positions = torch.arange(key_cache.size(2), device=value.device).view(1, -1)
        scores = scores.masked_fill(key_positions > query_positions, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, value_cache)
        context = context.transpose(1, 2).contiguous().view(batch, sequence_length, -1)
        return self.o_proj(context), key_cache, value_cache


class GatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, value):
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, heads, intermediate_size, capacity, epsilon=1e-5):
        super().__init__()
        self.input_norm = RMSNorm(hidden_size, epsilon)
        self.attention = CausalSelfAttention(hidden_size, heads, capacity)
        self.post_attention_norm = RMSNorm(hidden_size, epsilon)
        self.mlp = GatedMLP(hidden_size, intermediate_size)

    def forward(self, value, past_key=None, past_value=None):
        update, key_cache, value_cache = self.attention(
            self.input_norm(value), past_key, past_value
        )
        value = value + update
        value = value + self.mlp(self.post_attention_norm(value))
        return value, key_cache, value_cache


class DecoderOnlyLLM(nn.Module):
    def __init__(
        self,
        vocabulary_size=128,
        hidden_size=64,
        heads=4,
        layers=2,
        intermediate_size=128,
        capacity=32,
        epsilon=1e-5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hidden_size, heads, intermediate_size, capacity, epsilon)
                for _ in range(layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, epsilon)
        self.lm_head = nn.Linear(hidden_size, vocabulary_size, bias=False)

    def forward(self, token_ids, past_key_values=None):
        if past_key_values is None:
            past_key_values = [(None, None)] * len(self.layers)
        value = self.embedding(token_ids)
        next_cache = []
        for layer, (past_key, past_value) in zip(self.layers, past_key_values):
            value, key_cache, value_cache = layer(value, past_key, past_value)
            next_cache.append((key_cache, value_cache))
        return self.lm_head(self.final_norm(value)), next_cache
