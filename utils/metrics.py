import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

def calculate_bic(likelihood, beta, seq):
    k = len(beta)
    n = len(seq)
    
    bic = k * np.log(n) - 2 * likelihood
    return bic

def sample_sequence_from_model(model, model_func, beta, seq_length=10, start_word="goat"):
    sequence = [start_word]
    
    for i in range(seq_length - 1):
        prev_word = sequence[-1]
        if model_func == model.one_cue_static_global:
            prob_dist = np.array([np.exp(-model_func(beta, [word])) for word in model.freq.keys()])
        else:
            prob_dist = np.array([np.exp(-model_func(beta, [prev_word, word])) for word in model.freq.keys()])
        prob_dist /= prob_dist.sum()
        next_word = np.random.choice(list(model.freq.keys()), p=prob_dist)
        # next_word = list(model.freq.keys())[np.argmax(prob_dist)]
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

def get_persistance(sequence):
    pass

def plot_similarity(sequence):
    pass

def plot_RT(sequence):      # only applicable to true data
    pass

def get_repeats():
    pass

def calculate_bleu(generated_sequences, real_sequences):
    bleu_scores = {}
    for model_name, gen_seq in generated_sequences.items():
        score1 = sentence_bleu(real_sequences, gen_seq, weights=(1, 0, 0, 0))
        score2 = sentence_bleu(real_sequences, gen_seq, weights=(0, 1, 0, 0))
        score3 = sentence_bleu(real_sequences, gen_seq, weights=(0, 0, 1, 0))
        score4 = sentence_bleu(real_sequences, gen_seq, weights=(0, 0, 0, 1))

        bleu_scores[model_name] = {"1-gram": score1, "2-gram": score2, "3-gram": score3, "4-gram": score4}
    return bleu_scores

def calculate_rouge(generated_sequences, real_sequences):
    rouge_scores = {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for model_name, gen_seq in generated_sequences.items():
        gen_str = " ".join(gen_seq)
        scores = [scorer.score(ref, gen_str) for ref in real_sequences]
        avg_scores = {
            "rouge1": sum(score["rouge1"].fmeasure for score in scores) / len(scores),
            "rouge2": sum(score["rouge2"].fmeasure for score in scores) / len(scores),
            "rougeL": sum(score["rougeL"].fmeasure for score in scores)
        }
        rouge_scores[model_name] = avg_scores
    return rouge_scores