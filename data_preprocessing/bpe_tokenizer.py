import sentencepiece as spm

class BPETokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def encode(self, sentence):
        return self.sp.encode(sentence, out_type=int)
    
    def decode(self, token_ids):
        return self.sp.decode(token_ids)