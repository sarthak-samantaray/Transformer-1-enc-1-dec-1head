import torch

def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, device,max_length):
    src_indices = [src_tokenizer.sp.bos_id()] + src_tokenizer.encode(sentence) + [src_tokenizer.sp.eos_id()]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    model.eval()
    tgt_indices = [tgt_tokenizer.sp.bos_id()]
    
    with torch.no_grad():
        for _ in range(max_length):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            output = model(src_tensor, tgt_tensor)
            next_word_idx = output[0, -1].argmax().item()
            tgt_indices.append(next_word_idx)
            if next_word_idx == tgt_tokenizer.sp.eos_id():
                break
    
    return tgt_tokenizer.decode(tgt_indices[1:-1])  # Remove BOS and EOS tokens

