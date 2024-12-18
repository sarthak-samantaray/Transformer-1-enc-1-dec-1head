from predict import translate_sentence
import numpy as np
import os



def calculate_bleu_score(reference, hypothesis):
    """
    Calculate a simplified BLEU score (1-gram only for demonstration)
    """
    reference_words = reference.lower().split()
    hypothesis_words = hypothesis.lower().split()
    
    # Count matching words
    matches = sum(1 for hw in hypothesis_words if hw in reference_words)
    
    # Calculate precision
    if len(hypothesis_words) == 0:
        return 0
    return matches / len(hypothesis_words)

def test_model(model, test_pairs, src_vocab, tgt_vocab, device,max_len):
    model.eval()
    bleu_scores = []
    predictions = []
    
    print("\nTranslation Results:")
    print("-" * 60)
    print(f"{'English':<30}{'Predicted French':<30}{'Actual French'}")
    print("-" * 60)
    
    for src, tgt in test_pairs:
        # Get model prediction
        predicted = translate_sentence(model, src, src_vocab, tgt_vocab, device,max_len)
        predictions.append(predicted)
        
        # Calculate BLEU score
        bleu = calculate_bleu_score(tgt, predicted)
        bleu_scores.append(bleu)
        
        # Print results
        print(f"{src:<30}{predicted:<30}{tgt}")
    
    # Calculate and print metrics
    avg_bleu = np.mean(bleu_scores)
    print("\nMetrics:")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    
    # Print detailed analysis for each example
    print("\nDetailed Analysis:")
    print("-" * 60)
    for i, (src, tgt) in enumerate(test_pairs):
        print(f"\nExample {i+1}:")
        print(f"Source: {src}")
        print(f"Predicted: {predictions[i]}")
        print(f"Target: {tgt}")
        print(f"BLEU Score: {bleu_scores[i]:.4f}")