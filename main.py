import torch.nn as nn
import torch
import sentencepiece as spm
import os

from data_preprocessing.data_loader import prepare_data,create_batch , train_bpe_tokenizer_from_list
from data_preprocessing.vocabulary import create_vocabulary 
from data_preprocessing.bpe_tokenizer import BPETokenizer
from models.transformer import Transformeronlyselfattention
from train import train_transformer
from dataset import sentence_pairs , test_pairs
from test import test_model

from dotenv import load_dotenv
load_dotenv() 


EMBED_DIM = int(os.getenv('EMBED_DIM'))
FF_DIM = int(os.getenv('FF_DIM'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = int(os.getenv("MAX_LEN"))
model_prefix = 'bpe_model'

# Train the tokenizer
train_bpe_tokenizer_from_list(sentence_pairs, model_prefix)


sp = spm.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")

# Create vocabularies and prepare data
# src_vocab, tgt_vocab = create_vocabulary(sentence_pairs)
# src_tensors, tgt_tensors = prepare_data(sentence_pairs, src_vocab, tgt_vocab)
src_tokenizer = BPETokenizer("bpe_model.model")
tgt_tokenizer = BPETokenizer("bpe_model.model")

# prepare data
src_tensors, tgt_tensors = prepare_data(sentence_pairs, src_tokenizer, tgt_tokenizer)

model = Transformeronlyselfattention(
    src_vocab_size=src_tokenizer.sp.get_piece_size(),
    tgt_vocab_size=tgt_tokenizer.sp.get_piece_size(),
    embed_dim=EMBED_DIM,
    ff_dim=FF_DIM
)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
num_epochs = NUM_EPOCHS


# Create training data loader
train_data = list(create_batch(src_tensors, tgt_tensors))

# Train the model
train_transformer(model, train_data, optimizer, criterion, num_epochs, DEVICE)


# Test the model
print("Starting model testing...")
# test_model(model, test_pairs, src_vocab, tgt_vocab, DEVICE)
test_model(model, test_pairs, src_tokenizer, tgt_tokenizer, DEVICE,MAX_LEN)


