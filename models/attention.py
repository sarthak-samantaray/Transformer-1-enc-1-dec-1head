import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create Query, Key, Value matrices
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.scale = math.sqrt(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)  # Dropout after attention scores
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        return output
    

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create Query, Key, Value matrices
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.scale = math.sqrt(embed_dim)
        
    def forward(self, x, enc_output):
        Q = self.query(x)
        K = self.key(enc_output)
        V = self.value(enc_output)
        
        # Compute attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(attention, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        return output
