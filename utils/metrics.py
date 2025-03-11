import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

def calculate_bic(likelihood, beta, seq):
    k = len(beta)
    n = len(seq)
    
    bic = k * np.log(n) - 2 * likelihood
    return bic

def word_frequency_alignment(model, beta, real_sequences, num_samples=100):
    """
    Generate sequences and compare them to real sequences using summary statistics.
    """
    generated_sequences = [sample_sequence(model, beta, len(real_sequences[0])) for _ in range(num_samples)]
    
    # Example check: Compare word frequencies
    real_counts = get_counts([word for seq in real_sequences for word in seq])
    generated_counts = get_counts([word for seq in generated_sequences for word in seq])
    
    real_freq = np.array([real_counts[word] for word in model.freq.keys()])
    gen_freq = np.array([generated_counts.get(word, 0) for word in model.freq.keys()])
    
    plt.figure(figsize=(10,5))
    plt.scatter(real_freq, gen_freq, alpha=0.5)
    plt.xlabel("Real Word Frequency")
    plt.ylabel("Generated Word Frequency")
    plt.title("Posterior Predictive Check: Word Frequencies")
    plt.show()

def get_persistance(sequence):
    pass

def plot_similarity(sequence):
    pass

def plot_RT(sequence):
    pass

def get_repeats():
    pass

def calculate_bleu(generated_sequences, real_sequences):
    scores = []
    for gen_seq in generated_sequences:
        score1 = sentence_bleu(real_sequences, gen_seq, weights=(1, 0, 0, 0))
        score2 = sentence_bleu(real_sequences, gen_seq, weights=(0, 1, 0, 0))
        score3 = sentence_bleu(real_sequences, gen_seq, weights=(0, 0, 1, 0))
        score4 = sentence_bleu(real_sequences, gen_seq, weights=(0, 0, 0, 1))
        scores.append([score1, score2, score3, score4])
    return dict(zip(["bleu1", "bleu2", "bleu3", "bleu4"], np.round(np.mean(scores, axis=0), 2).tolist()))

# def calculate_rouge(generated_sequences, real_sequences):
#     scores = []
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     for gen_seq in generated_sequences:
#         scores.append([scorer.score(ref, gen_seq) for ref in real_sequences])
#     return dict(zip(['rouge1', 'rouge2', 'rougeL'], np.round(np.mean(scores, axis=0), 2).tolist()))

def calculate_rouge(generated_sequences, real_sequences):
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for gen_seq in generated_sequences:
        rouge_scores = [
            max(scorer.score(ref, gen_seq)['rouge1'].fmeasure for ref in real_sequences),
            max(scorer.score(ref, gen_seq)['rouge2'].fmeasure for ref in real_sequences),
            max(scorer.score(ref, gen_seq)['rougeL'].fmeasure for ref in real_sequences)
        ]
        scores.append(rouge_scores)

    mean_scores = np.round(np.mean(scores, axis=0), 2).tolist()
    return dict(zip(['rouge1', 'rouge2', 'rougeL'], mean_scores))