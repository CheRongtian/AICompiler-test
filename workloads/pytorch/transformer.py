import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        
        return out, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_k = d_model // n_heads # 每个头的维度
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        out, attn = self.attention(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        out = self.fc(out)
        out = self.dropout(out)

        return self.norm(out+q), attn

class FeedFroward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        out = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        return self.norm(out+x)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedFroward(d_model, d_ff, dropout)

    def forward(self, src, src_mask=None):
        out, _ = self.self_attn(src, src, src, src_mask)
        out = self.ffn(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedFroward(d_model, d_ff, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        out, _ = self.cross_attn(out, memory, memory, memory_mask)
        out = self.ffn(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,num_layers,d_ff,dropout=0.1,max_len=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        
        out = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        out = self.pos_encoding(out)

        for layer in self.layers:
            out = layer(out, src_mask) # self_attn + ffn

        return out
    
class Decoder(nn.Module):
    def __init__(self, vocab_size,d_model,n_heads,num_layers,d_ff,dropout=0.1,max_len=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_task=None):
        out = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        out = self.pos_encoding(out)

        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_task)

        return self.fc_out(out)

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 d_model=512,
                 n_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_len=5000
                 ):
        super().__init__()
        
        self.encoder = Encoder(
            src_vocab, d_model, n_heads, num_encoder_layers, d_ff, dropout, max_len
        )
        
        self.decoder = Decoder(
            tgt_vocab, d_model, n_heads, num_decoder_layers, d_ff, dropout, max_len
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, tgt_mask, memory_mask)
        
        return out
        
def generate_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask==0
