import torch
import torch.nn as nn
from .encoder import EncoderLayer
from .decoder import DecoderLayer
import math

class Transformeronlyselfattention(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=512, ff_dim=512):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.positional_encoding = self.create_positional_encoding(5000, embed_dim)
        
        # Encoder and Decoder
        self.encoder = EncoderLayer(embed_dim, ff_dim)
        self.decoder = DecoderLayer(embed_dim, ff_dim)
        
        # Output layer
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)
        
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, src, tgt):
        # Embedding + Positional Encoding
        src_embedded = self.src_embedding(src) + self.positional_encoding[:, :src.shape[1], :].to(src.device)
        tgt_embedded = self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.shape[1], :].to(tgt.device)
        
        # Encoder
        enc_output = self.encoder(src_embedded)
        
        # Decoder
        dec_output = self.decoder(tgt_embedded, enc_output)
        
        # Output layer
        output = self.output_layer(dec_output)
        return output
