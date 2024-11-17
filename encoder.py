#let create decoder only model

import torch
import torch.nn as nn
import torch.nn.functional as F

class Magic(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(Magic, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_ff = d_ff

        # Multi-Head Attention components
        self.Q_linear = nn.Linear(d_model, d_model)
        self.K_linear = nn.Linear(d_model, d_model)
        self.V_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)

        # Feedforward Network (FFN)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Compute Q, K, V
        Q = self.Q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.K_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.V_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores: Q @ K.T and apply scaling
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5  # Scale attention scores 
        # Apply softmax to normalized scores
        attn = F.softmax(attn, dim=-1)
        # Compute the attention output
        output = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(output)
        # Add & Normalize
        x = self.norm1(x + output)
        # Feed-Forward Network (FFN)
        ffn_out = F.relu(self.ffn1(x))
        ffn_out = self.ffn2(ffn_out)

        # Add & Normalize
        x = self.norm2(x + ffn_out)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, batch_size, d_model=80, n_heads=2, d_ff=2048, max_length=10000):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_length = max_length
        self.emb = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_length, d_model)
        self.magics = nn.ModuleList([Magic(d_model, n_heads, d_ff) for _ in range(3)])
        self.output_layer = nn.Linear(d_model, vocab_size)


    def create_positional_encoding(self, seq_len, embed_dim):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        dim = torch.arange(embed_dim, dtype=torch.float).unsqueeze(0)
        angles = position / (10000 ** (dim / embed_dim))
        pos_encoding = torch.zeros(seq_len, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angles[:, 1::2])
        return pos_encoding

    def forward(self, inputs):
        # Adjust for potential singleton middle dimension in testing
        if inputs.dim() == 3 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # Squeeze the middle dimension if it’s just [batch_size, 1, seq_len]

        batch_size, seq_len = inputs.size()
        embs = self.emb(inputs)  # Shape: [batch_size, seq_len, d_model]
        #print(f"embs shape: {embs.shape}")

        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1).to(embs.device)
        #print(f"pos_enc shape: {pos_enc.shape}")

        embs = embs + pos_enc
        #print(f"embs shape after adding pos_enc: {embs.shape}")

        for magic in self.magics:
            embs = magic(embs)

        logits = self.output_layer(embs)
        return logits


def main():
    decoder = Encoder(vocab_size=100, batch_size=12)
    inputs = torch.randint(0, 100, (12, 10)) # Batch size of 12, sequence length of 10, the 0, 10 define the bound for random number generation
    print("the input shape is", inputs.shape)
    logits = decoder(inputs)
    print("the output shape of logits is", logits.shape)
    print("the output type of logits is", type(logits))

    # the input shape is torch.Size([12, 10])
    
    # embs shape: torch.Size([12, 10, 80])
    # pos_enc shape: torch.Size([12, 10, 80])
    # embs shape after adding pos_enc: torch.Size([12, 10, 80])
    # the output shape of logits is torch.Size([12, 10, 100])

if __name__ == "__main__":
    main()