import torch.nn as nn
from .attention import SelfAttention

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)  # Dropout for residual connection
        self.dropout2 = nn.Dropout(dropout)  # Dropout for FFN residual connection
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout within FFN
            nn.Linear(ff_dim, embed_dim)
        )
        
    def forward(self, x):
        # Self Attention with residual connection and layer norm
        attention_output = self.self_attention(x)
        x = self.norm1(x + self.dropout1(attention_output))  # Apply dropout
        
        # Feed forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))  # Apply dropout
        return x