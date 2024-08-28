import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# register buffer in Pytorch ->
# If you have parameters in your model, which should be saved and restored in the state_dict,
# but not trained by the optimizer, you should register them as buffers.
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, emb_dim):
        """
        Args:
            max_seq_len: int, maximum sequence length
            emb_dim: int, embedding dimension
        """
        super().__init__()
        self.emb_dim = emb_dim

        pe = torch.zeros(max_seq_len, emb_dim)
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / emb_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / emb_dim)))

        pe = pe.unsqueeze(0)  # insert a dimension at the 0th position
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, seq_len, emb_dim)
        """
        x = x * math.sqrt(self.emb_dim)  # make embeddings relatively larger
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        """
        Args:
            d_model: int, dimension of the model/embedding vector output
            num_heads: int, number of self-attention heads
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension of each head's key, query, value
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # key, query, value weight matrices, 64 x 64
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # output layer

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        attention score is usually with shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Calculate the attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Multiply by value to get the output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multi-heads to get the original shape
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: torch.Tensor, query tensor of shape (batch_size, seq_len, d_model)
            K: torch.Tensor, key tensor of shape (batch_size, seq_len, d_model)
            V: torch.Tensor, value tensor of shape (batch_size, seq_len, d_model)
            mask: torch.Tensor, mask tensor of shape (batch_size, seq_len)
        """
        # Linearly project the queries, keys, and values
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split the queries, keys, and values into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine the multi-heads
        combined_attn_output = self.combine_heads(attn_output)

        # Linearly project the output
        output = self.W_o(combined_attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        """
        Args:
            d_model: int, dimension of the model/embedding vector output
            d_ff: int, dimension of the feed-forward layer
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Two linear transformations with ReLU activation in between
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, seq_len, d_model)
            mask: torch.Tensor, mask tensor of shape (batch_size, seq_len)
        """
        # Apply self-attention
        attn_output = self.self_attn(x, x, x, mask)  # Q, K, V are all x
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # residual connection
        x = self.norm1(x)

        # Apply feed-forward network
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output
        x = self.norm2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = self.norm1(x)

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = self.norm2(x)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output
        x = self.norm3(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_heads,
        num_layers,
        d_model,
        d_ff,
        max_seq_len,
        dropout=0.0,
    ):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        """
        attention score is usually with shape (batch_size, num_heads, seq_len, seq_len)
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # (batch_size, 1, tgt_len, 1)
        seq_len = tgt.size(1)
        nopeak_mask = (
            1
            - torch.triu(
                torch.ones(1, seq_len, seq_len, device=tgt_mask.device), diagonal=1
            )
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src = self.positional_encoding(self.encoder_embedding(src))
        tgt = self.positional_encoding(self.decoder_embedding(tgt))

        for encoder in self.encoder_blocks:
            src = encoder(src, src_mask)

        for decoder in self.decoder_blocks:
            tgt = decoder(tgt, src, src_mask, tgt_mask)

        output = self.fc(tgt)
        return output
