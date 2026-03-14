import numpy as np

def build_vocab(corpus):
    vocab = set()
    for word in corpus:
        vocab.add(word)

    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for i, word in enumerate(vocab)}
    return word_to_index, index_to_word

def sample_negative_samples(vocab_size, center_word, num_negative_samples=10):
    negative_samples = []
    for i in range(num_negative_samples):
        neg = np.random.randint(0, vocab_size)
        while neg == center_word: #avoid to smaple center
            neg = np.random.randint(0, vocab_size)
        negative_samples.append(neg)
    return negative_samples

def generate_pairs(tokens, window_size):
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(- window_size, window_size + 1):
            if j == 0: #center word
                continue
            if i+j <0 or i+j >= len(tokens):  # out of bounds (center is near boundery)
                continue

            context = tokens[i+j]
            pairs.append((center, context))
    return pairs