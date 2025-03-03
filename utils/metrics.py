import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

def calculate_bic(likelihood, beta, seq):
    """
    Compute BIC score for a given model.
    """
    k = len(beta)
    n = len(seq)
    
    bic = k * np.log(n) - 2 * likelihood
    return bic

# def sample_sequence(model, beta, seq_length=20, start_word="dog"):
#     """
#     Generate a sequence of words given a trained model.
#     """
#     sequence = [start_word]
#     for _ in range(seq_length - 1):
#         prev_word = sequence[-1]
#         prob_dist = np.array([
#             pow(model.freq[word], beta[0]) * pow(model.sim_mat[prev_word][word], beta[1])
#             for word in model.freq.keys()
#         ])
#         prob_dist /= prob_dist.sum()  # Normalize probabilities
#         next_word = np.random.choice(list(model.freq.keys()), p=prob_dist)
#         sequence.append(str(next_word))
#     return sequence

def sample_sequence_from_model(model, model_func, beta, seq_length=10, start_word="dog"):
    """
    Generate a sequence of words using a specific model function.
    
    Args:
        model: The Hills model instance.
        model_func: The model function to sample from.
        beta: Optimized beta parameters.
        seq_length: Length of the sequence to generate.
        start_word: Initial word in the sequence.
    
    Returns:
        Generated sequence.
    """
    sequence = [start_word]
    
    for _ in range(seq_length - 1):
        prev_word = sequence[-1]
        
        prob_dist = np.array([np.exp(-model_func(beta, [prev_word, word])) for word in model.freq.keys()])
        
        prob_dist /= prob_dist.sum()

        next_word = np.random.choice(list(model.freq.keys()), p=prob_dist)
        sequence.append(str(next_word))
    
    return sequence

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

def calculate_bleu_scores(real_sequences, generated_sequences):
    """
    Compute BLEU scores between real and generated sequences.
    """
    bleu_scores = []
    for real, gen in zip(real_sequences, generated_sequences):
        score = sentence_bleu([real], gen)  # Treat real as reference
        bleu_scores.append(score)
    
    return np.mean(bleu_scores), np.std(bleu_scores)