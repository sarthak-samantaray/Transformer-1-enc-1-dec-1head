import torch
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import os

def prepare_data(sentence_pairs, src_tokenizer, tgt_tokenizer):
    src_tensors = []
    tgt_tensors = []
    
    for src, tgt in sentence_pairs:
        src_indices = [src_tokenizer.sp.bos_id()] + src_tokenizer.encode(src) + [src_tokenizer.sp.eos_id()]
        tgt_indices = [tgt_tokenizer.sp.bos_id()] + tgt_tokenizer.encode(tgt) + [tgt_tokenizer.sp.eos_id()]
        
        src_tensors.append(torch.tensor(src_indices))
        tgt_tensors.append(torch.tensor(tgt_indices))
    
    return src_tensors, tgt_tensors


def train_bpe_tokenizer_from_list(sentence_pairs, model_prefix, vocab_size=275):
    """
    Train a BPE tokenizer using SentencePiece from a list of sentence pairs.

    Args:
    - sentence_pairs (list of tuples): List of (source, target) sentence pairs.
    - model_prefix (str): Prefix for the generated tokenizer model files.
    - vocab_size (int): Vocabulary size for the tokenizer.
    """
    # Combine source and target sentences into one text
    sentences = [src + '' + tgt for src, tgt in sentence_pairs]
    
    # Write combined sentences to a temporary file
    temp_file = 'temp_training_data.txt'
    with open(temp_file, 'w') as f:
        f.write("\n".join(sentences))

    # Now, use this file to train the tokenizer
    try:
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",  # Use BPE model
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,  # Special token IDs
            user_defined_symbols=["<START>", "<END>"],  # Optional additional tokens
        )
        
        # Confirm that the model files were created
        if not os.path.exists(f"{model_prefix}.model") or not os.path.exists(f"{model_prefix}.vocab"):
            raise FileNotFoundError(f"Model files '{model_prefix}.model' or '{model_prefix}.vocab' were not created.")
        
        print(f"Model training completed successfully. Files saved as {model_prefix}.model and {model_prefix}.vocab")
        
    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")

def create_batch(src_tensors, tgt_tensors, batch_size=4):
    for i in range(0, len(src_tensors), batch_size):
        src_batch = src_tensors[i:i+batch_size]
        tgt_batch = tgt_tensors[i:i+batch_size]
        
        src_padded = pad_sequence(src_batch, batch_first=True)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True)
        
        yield src_padded, tgt_padded
