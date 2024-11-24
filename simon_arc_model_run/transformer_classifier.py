import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len=500):
        super(PositionalEncoding, self).__init__()

        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_length, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, src_vocab_size=528, num_classes=10, d_model=128, nhead=8, num_layers=6):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model

        # Embedding and positional encoding
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = src.transpose(0, 1)  # [seq_length, batch_size]
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        output = self.transformer_encoder(src_emb)

        # Pool over the sequence dimension (e.g., take the mean)
        output = output.mean(dim=0)

        # Final classification layer
        output = self.fc_out(output)
        return output  # [batch_size, num_classes]
