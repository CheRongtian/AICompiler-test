import argparse
import math
import time

import torch
import torch.nn as nn

from KVCache import MultiHeadAttentionKVCache, calculate_kv_cache_size
from transformer import Encoder, FeedFroward, MultiHeadAttention, Transformer, generate_mask


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def tensor_size_mb(tensor):
    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def total_kv_cache_mb(layer_past_key_values):
    return sum(calculate_kv_cache_size(past) for past in layer_past_key_values if past is not None)


def decoder_kv_tensor_mb(batch, seq_len, d_model, decoder_layers, dtype=torch.float32):
    bytes_per_value = torch.tensor([], dtype=dtype).element_size()
    total_bytes = decoder_layers * 2 * batch * seq_len * d_model * bytes_per_value
    return total_bytes / (1024 * 1024)


def zero_decoder_self_attn_biases(model):
    for layer in model.decoder.layers:
        layer.self_attn.W_q.bias.data.zero_()
        layer.self_attn.W_k.bias.data.zero_()
        layer.self_attn.W_v.bias.data.zero_()
        layer.self_attn.fc.bias.data.zero_()


def copy_weights_to_cached_transformer(source_model, cached_model):
    cached_model.encoder.load_state_dict(source_model.encoder.state_dict())
    cached_model.decoder.embedding.load_state_dict(source_model.decoder.embedding.state_dict())
    cached_model.decoder.fc_out.load_state_dict(source_model.decoder.fc_out.state_dict())
    cached_model.decoder.pos_encoding.copy_(source_model.decoder.pos_encoding.pe)

    for source_layer, cached_layer in zip(source_model.decoder.layers, cached_model.decoder.layers):
        cached_layer.self_attn.q_proj.weight.data.copy_(source_layer.self_attn.W_q.weight.data)
        cached_layer.self_attn.k_proj.weight.data.copy_(source_layer.self_attn.W_k.weight.data)
        cached_layer.self_attn.v_proj.weight.data.copy_(source_layer.self_attn.W_v.weight.data)
        cached_layer.self_attn.o_proj.weight.data.copy_(source_layer.self_attn.fc.weight.data)
        cached_layer.self_attn_norm.load_state_dict(source_layer.self_attn.norm.state_dict())
        cached_layer.cross_attn.load_state_dict(source_layer.cross_attn.state_dict())
        cached_layer.ffn.load_state_dict(source_layer.ffn.state_dict())


class CachedDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionKVCache(d_model, n_heads, use_kv=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedFroward(d_model, d_ff, dropout)

    def forward(self, tgt_step, memory, past_key_value=None, memory_mask=None):
        past_key = None
        past_value = None
        if past_key_value is not None:
            past_key, past_value = past_key_value

        self_attn_out, new_past_key_value = self.self_attn(
            tgt_step,
            tgt_step,
            tgt_step,
            past_key=past_key,
            past_value=past_value,
        )
        out = self.self_attn_norm(tgt_step + self.self_attn_dropout(self_attn_out))
        out, _ = self.cross_attn(out, memory, memory, memory_mask)
        out = self.ffn(out)
        return out, new_past_key_value


class CachedDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", self._build_positional_encoding(d_model, max_len))
        self.layers = nn.ModuleList(
            [CachedDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _build_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward_step(self, tgt_step, memory, layer_past_key_values=None, step=0, memory_mask=None):
        if layer_past_key_values is None:
            layer_past_key_values = [None] * len(self.layers)

        out = self.embedding(tgt_step) * math.sqrt(self.embedding.embedding_dim)
        out = out + self.pos_encoding[:, step : step + tgt_step.size(1), :]

        new_layer_past_key_values = []
        for layer, past_key_value in zip(self.layers, layer_past_key_values):
            out, new_past_key_value = layer(out, memory, past_key_value, memory_mask)
            new_layer_past_key_values.append(new_past_key_value)

        return self.fc_out(out), new_layer_past_key_values


class TransformerWithKVCache(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_heads, num_encoder_layers, d_ff, dropout, max_len)
        self.decoder = CachedDecoder(tgt_vocab, d_model, n_heads, num_decoder_layers, d_ff, dropout, max_len)

    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    def decode_step(self, tgt_step, memory, layer_past_key_values=None, step=0, memory_mask=None):
        return self.decoder.forward_step(tgt_step, memory, layer_past_key_values, step, memory_mask)


@torch.no_grad()
def benchmark_without_kv(args, device):
    model = Transformer(
        args.src_vocab,
        args.tgt_vocab,
        args.d_model,
        args.n_heads,
        args.encoder_layers,
        args.decoder_layers,
        args.d_ff,
        args.dropout,
        args.max_len,
    ).to(device)
    model.eval()

    src = torch.randint(0, args.src_vocab, (args.batch, args.src_len), device=device)
    generated = torch.randint(0, args.tgt_vocab, (args.batch, 1), device=device)
    memory = model.encoder(src)

    start = time.perf_counter()
    for _ in range(args.decode_steps):
        tgt_mask = generate_mask(generated.size(1)).to(device)
        logits = model.decoder(generated, memory, tgt_mask=tgt_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    elapsed = time.perf_counter() - start

    output_mb = tensor_size_mb(generated)
    recompute_kv_mb = decoder_kv_tensor_mb(
        args.batch,
        generated.size(1),
        args.d_model,
        args.decoder_layers,
    )
    return elapsed, output_mb, recompute_kv_mb, generated.shape


@torch.no_grad()
def benchmark_with_kv(args, device):
    model = TransformerWithKVCache(
        args.src_vocab,
        args.tgt_vocab,
        args.d_model,
        args.n_heads,
        args.encoder_layers,
        args.decoder_layers,
        args.d_ff,
        args.dropout,
        args.max_len,
    ).to(device)
    model.eval()

    src = torch.randint(0, args.src_vocab, (args.batch, args.src_len), device=device)
    next_token = torch.randint(0, args.tgt_vocab, (args.batch, 1), device=device)
    memory = model.encode(src)
    layer_past_key_values = None

    start = time.perf_counter()
    for step in range(args.decode_steps):
        logits, layer_past_key_values = model.decode_step(
            next_token,
            memory,
            layer_past_key_values=layer_past_key_values,
            step=step,
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    elapsed = time.perf_counter() - start

    kv_mb = total_kv_cache_mb(layer_past_key_values)
    return elapsed, kv_mb, next_token.shape


@torch.no_grad()
def check_kv_correctness(args, device):
    source_model = Transformer(
        args.src_vocab,
        args.tgt_vocab,
        args.d_model,
        args.n_heads,
        args.encoder_layers,
        args.decoder_layers,
        args.d_ff,
        args.dropout,
        args.max_len,
    ).to(device)
    cached_model = TransformerWithKVCache(
        args.src_vocab,
        args.tgt_vocab,
        args.d_model,
        args.n_heads,
        args.encoder_layers,
        args.decoder_layers,
        args.d_ff,
        args.dropout,
        args.max_len,
    ).to(device)

    zero_decoder_self_attn_biases(source_model)
    copy_weights_to_cached_transformer(source_model, cached_model)
    source_model.eval()
    cached_model.eval()

    check_steps = min(args.correctness_steps, args.max_len)
    src = torch.randint(0, args.src_vocab, (args.batch, args.src_len), device=device)
    tgt = torch.randint(0, args.tgt_vocab, (args.batch, check_steps), device=device)

    source_memory = source_model.encoder(src)
    cached_memory = cached_model.encode(src)
    layer_past_key_values = None
    max_abs_diff = 0.0
    passed = True

    for step in range(check_steps):
        prefix = tgt[:, : step + 1]
        tgt_mask = generate_mask(prefix.size(1)).to(device)
        source_logits = source_model.decoder(prefix, source_memory, tgt_mask=tgt_mask)[:, -1, :]
        cached_logits, layer_past_key_values = cached_model.decode_step(
            tgt[:, step : step + 1],
            cached_memory,
            layer_past_key_values=layer_past_key_values,
            step=step,
        )
        cached_logits = cached_logits[:, -1, :]
        diff = (source_logits - cached_logits).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        if not torch.allclose(source_logits, cached_logits, atol=args.atol, rtol=args.rtol):
            passed = False

    return passed, max_abs_diff, check_steps


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer decoding with/without KV cache.")
    parser.add_argument("--mode", choices=("both", "kv", "no-kv"), default="both")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--src-len", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--src-vocab", type=int, default=10000)
    parser.add_argument("--tgt-vocab", type=int, default=10000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-len", type=int, default=5000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--use-kv", type=str_to_bool, default=None, help="shell-style switch: true -> kv, false -> no-kv")
    parser.add_argument("--check-correctness", type=str_to_bool, default=True)
    parser.add_argument("--correctness-steps", type=int, default=16)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    if args.use_kv is True:
        args.mode = "kv"
    elif args.use_kv is False:
        args.mode = "no-kv"

    assert args.d_model % args.n_heads == 0, "--d-model must be divisible by --n-heads"
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    print(f"device: {device}")
    print(
        "config: "
        f"batch={args.batch}, src_len={args.src_len}, decode_steps={args.decode_steps}, "
        f"d_model={args.d_model}, heads={args.n_heads}, decoder_layers={args.decoder_layers}"
    )

    no_kv_result = None
    kv_result = None

    if args.mode in ("both", "no-kv"):
        no_kv_result = benchmark_without_kv(args, device)
        print(
            f"no-kv time: {no_kv_result[0]:.4f}s | "
            f"persistent generated tokens: {no_kv_result[1]:.4f} MB | "
            f"final-step recomputed K/V estimate: {no_kv_result[2]:.4f} MB"
        )

    if args.mode in ("both", "kv"):
        kv_result = benchmark_with_kv(args, device)
        print(f"kv-cache time: {kv_result[0]:.4f}s | kv cache: {kv_result[1]:.4f} MB")

    if no_kv_result is not None and kv_result is not None:
        speedup = no_kv_result[0] / kv_result[0] if kv_result[0] > 0 else float("inf")
        print(f"speedup: {speedup:.2f}x")

    if args.check_correctness:
        passed, max_abs_diff, check_steps = check_kv_correctness(args, device)
        print(
            f"correctness allclose: {passed} | "
            f"steps={check_steps}, max_abs_diff={max_abs_diff:.6g}, atol={args.atol}, rtol={args.rtol}"
        )


if __name__ == "__main__":
    main()
