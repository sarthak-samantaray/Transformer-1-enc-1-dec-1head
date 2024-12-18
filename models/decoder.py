import torch.nn as nn
from .attention import SelfAttention , CrossAttention

# class DecoderLayer(nn.Module):
#     def __init__(self, embed_dim, ff_dim, dropout=0.1):
#         super().__init__()
#         self.self_attention = SelfAttention(embed_dim, dropout)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
        
#         self.ff = nn.Sequential(
#             nn.Linear(embed_dim, ff_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_dim, embed_dim)
#         )
        
#     def forward(self, x, enc_output):
#         # Self Attention
#         attention_output = self.self_attention(x)
#         x = self.norm1(x + self.dropout1(attention_output))
        
#         # Feed forward
#         ff_output = self.ff(x)
#         x = self.norm2(x + self.dropout2(ff_output))
#         return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim)
        self.cross_attention = CrossAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
    def forward(self, x, enc_output):
        # Self Attention
        attention_output = self.self_attention(x)
        x = self.norm1(x + attention_output)
        
        # Cross Attention
        cross_attention_output = self.cross_attention(x, enc_output)
        x = self.norm2(x + cross_attention_output)
        
        # Feed forward
        ff_output = self.ff(x)
        x = self.norm3(x + ff_output)
        return x
