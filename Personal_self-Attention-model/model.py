import torch
import torch.nn as nn

'''
    From the original "Attention is All You Need" paper:

        PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
        PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        '''
            split the embedding dimension (d_model) into multiple heads (num_heads).
            Each head will have a dimension of d_k = d_model / num_heads.

            For each head, the weight matrices shall bear the shape (d_model, d_k) and
            by representing them as a single large matrix, we have (d_model, d_model).
        '''
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch_size, num_heads, seq_len, d_k)

        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = attn_weights @ V
        # attn_weights: (batch_size, num_heads, seq_len), attn_output: (batch_size, num_heads, seq_len, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        # attn_output: (batch_size, seq_len, d_model)

        return self.W_o(attn_output)
    
class mlp(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = mlp(d_model, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x
    

class SemanticClassifier(nn.Module):
    def __init__(
            self,
            num_classes,
            vocab_size,
            max_len,

            d_model = 128,
            n_heads = 4,
            ff_dim = 4 * 128,
            n_layers = 2,
            dropout=0.1
        ):
        super().__init__()        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.transformers = nn.ModuleList(
            [Transformer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)

        for transformer in self.transformers:
            x = transformer(x)

        cls_rep = x[:, 0, :]  # Take the representation of the [CLS] token
        cls_rep = self.dropout(cls_rep)
        logits = self.fc_out(cls_rep)

        return logits

        
        


