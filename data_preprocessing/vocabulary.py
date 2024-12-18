def create_vocabulary(sentence_pairs):
    src_vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2}
    tgt_vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2}
    
    for src, tgt in sentence_pairs:
        for word in src.lower().split():
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)
        for word in tgt.lower().split():
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)
    
    return src_vocab, tgt_vocab
