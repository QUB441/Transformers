#let's do encoder, decoder, and then the transition in between

#encoder needs to return the results of the K and V in the same emb dimension


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

        print("the shape of q", Q.shape)
        print("the shape of k", K.shape)
        print("the shape of v", V.shape)

        # Compute attention scores: Q @ K.T and apply scaling
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5  # Scale attention scores 
        # Apply softmax to normalized scores
        attn = F.softmax(attn, dim=-1)
        print("the shape of attention is", attn.shape)
        # Compute the attention output
        output = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(output)
        print("the shape of output is", output.shape)
        
        
        # Add & Normalize
        x = self.norm1(x + output)
        # Feed-Forward Network (FFN)
        ffn_out = F.relu(self.ffn1(x))
        ffn_out = self.ffn2(ffn_out)

        # Add & Normalize
        x = self.norm2(x + ffn_out)
        print("the shape of x is", x.shape)

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
        pos_encoding = torch.zeros(seq_len, embed_dim, device=self.emb.weight.device)
        pos_encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angles[:, 1::2])
        return pos_encoding

    def forward(self, inputs):
        # Adjust for potential singleton middle dimension in testing
        if inputs.dim() == 3 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # Squeeze the middle dimension if it’s just [batch_size, 1, seq_len]

        batch_size, seq_len = inputs.size()
        embs = self.emb(inputs)  # Shape: [batch_size, seq_len, d_model]
        print(f"embs shape: {embs.shape}")

        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1).to(embs.device)
        print(f"pos_enc shape: {pos_enc.shape}")

        embs = embs + pos_enc
        print(f"embs shape after adding pos_enc: {embs.shape}")

        for magic in self.magics:
            embs = magic(embs)

        #logits = self.output_layer(embs)
        return embs

##### let's add decoder

class decoderMagic(nn.Module):
    def __init__(self, d_model,encoder_d_model, n_heads, d_ff):
        super(decoderMagic, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_ff = d_ff

        # Multi-Head Attention components
        self.Q_linear = nn.Linear(d_model, encoder_d_model) #how to pass this?
        self.K_linear = nn.Linear(d_model, encoder_d_model)
        self.V_linear = nn.Linear(d_model, encoder_d_model)

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
        # Create a mask to prevent attending to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
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


class TransDecoder(nn.Module):
    def __init__(self, vocab_size, batch_size, d_model=80, n_heads=2, d_ff=2048, max_length=10000):
        super(TransDecoder, self).__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_length = max_length
        self.emb = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_length, d_model)
        self.magics = nn.ModuleList([Magic(d_model, n_heads, d_ff) for _ in range(3)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.self_attention = nn.MultiheadAttention(d_model, n_heads)  # Adjust heads as necessary
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads)  # Cross-attention layer


    def create_positional_encoding(self, seq_len, embed_dim):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        dim = torch.arange(embed_dim, dtype=torch.float).unsqueeze(0)
        angles = position / (10000 ** (dim / embed_dim))
        pos_encoding = torch.zeros(seq_len, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angles[:, 1::2])
        return pos_encoding

    def forward(self, encoder_embs, decoder_inputs):
        # Get embeddings for decoder inputs
        batch_size, seq_len = decoder_inputs.size()
        embs = self.emb(decoder_inputs)  # Shape: [batch_size, seq_len, d_model]

        # Apply positional encoding to decoder embeddings
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1).to(embs.device)
        embs = embs + pos_enc

        # Apply Magic layers
        for magic in self.magics:
            embs = magic(embs)
        
        # Apply cross-attention
        embs, _ = self.cross_attention(embs, encoder_embs, encoder_embs)  # Cross-attention between encoder and decoder

        # # Cross-Attention between Encoder and Decoder embeddings
        # embs = self.cross_attention(encoder_embs, embs)


        return embs
    # def forward(self, inputs):
    #     # Adjust for potential singleton middle dimension in testing
    #     if inputs.dim() == 3 and inputs.size(1) == 1:
    #         inputs = inputs.squeeze(1)  # Squeeze the middle dimension if it’s just [batch_size, 1, seq_len]

    #     batch_size, seq_len = inputs.size()
    #     embs = self.emb(inputs)  # Shape: [batch_size, seq_len, d_model]
    #     #print(f"embs shape: {embs.shape}")

    #     pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1).to(embs.device)
    #     #print(f"pos_enc shape: {pos_enc.shape}")

    #     embs = embs + pos_enc
    #     #print(f"embs shape after adding pos_enc: {embs.shape}")

    #     for magic in self.magics:
    #         embs = magic(embs)

   
    #     return embs
    
def cross_attention(self, encoder_embs, decoder_embs):
        """
        Perform cross-attention between encoder embeddings and decoder embeddings.
        
        Args:
        - encoder_embs: Tensor of shape [batch_size, seq_len_enc, d_model].
        - decoder_embs: Tensor of shape [batch_size, seq_len_dec, d_model].
        
        Returns:
        - Updated decoder embeddings: Tensor of shape [batch_size, seq_len_dec, d_model].
        """
        batch_size, seq_len_enc, d_model = encoder_embs.size()
        _, seq_len_dec, _ = decoder_embs.size()

        # Compute cross-attention scores
        cross_attn = torch.matmul(decoder_embs, encoder_embs.transpose(-2, -1))  # [batch_size, seq_len_dec, seq_len_enc]
        cross_attn = cross_attn / (self.d_k ** 0.5)  # Scale scores
        cross_attn = F.softmax(cross_attn, dim=-1)  # Normalize scores
        
        # Weighted sum of encoder embeddings
        output = torch.matmul(cross_attn, encoder_embs)  # [batch_size, seq_len_dec, d_model]
        
        # Apply linear projection
        output = self.out_linear(output)  # [batch_size, seq_len_dec, d_model]
        
        # Add & Normalize with residual connection
        x = self.norm1(decoder_embs + output)  # Residual connection
        
        # Feedforward network
        ffn_out = F.relu(self.ffn1(x))  # First FFN layer
        ffn_out = self.ffn2(ffn_out)    # Second FFN layer
        
        # Add & Normalize with residual connection
        x = self.norm2(x + ffn_out)  # Residual connection
        
        return x


def main():
    # Define parameters
    vocab_size = 100
    batch_size = 12
    seq_len_enc = 10
    seq_len_dec = 8
    d_model = 80

    # Initialize Encoder and Decoder
    encoder = Encoder(vocab_size=vocab_size, batch_size=batch_size, d_model=d_model)
    decoder = TransDecoder(vocab_size=vocab_size, batch_size=batch_size, d_model=d_model)

    # Generate random inputs for encoder and decoder
    encoder_inputs = torch.randint(0, vocab_size, (batch_size, seq_len_enc))  # Encoder input
    decoder_inputs = torch.randint(0, vocab_size, (batch_size, seq_len_dec))  # Decoder input

    # Pass encoder inputs through the encoder
    print("Testing Encoder...")
    encoder_outputs = encoder(encoder_inputs)
    print(f"Encoder output shape: {encoder_outputs.shape}")  # Should be [batch_size, seq_len_enc, d_model]

    # Pass decoder inputs through the decoder
    print("\nTesting Decoder...")
    decoder_embs = decoder(encoder_outputs, decoder_inputs)  # Pass encoder outputs to decoder
    print(f"Decoder embeddings shape: {decoder_embs.shape}")  # Should be [batch_size, seq_len_dec, d_model]

    # Final assertion to confirm all shapes match expectations
    assert encoder_outputs.shape == (batch_size, seq_len_enc, d_model), "Encoder output shape mismatch!"
    assert decoder_embs.shape == (batch_size, seq_len_dec, d_model), "Decoder embeddings shape mismatch!"

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    main()