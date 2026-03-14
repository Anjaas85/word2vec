import numpy as np

from src.word2vec import Word2Vec
from src.utils import build_vocab, sample_negative_samples, generate_pairs

def train(text, embedding_dim=100, learning_rate=0.001, window_size=5, epochs=2,num_negatives = 10):

    word_to_index, index_to_word = build_vocab(text)
    vocab_size = len(word_to_index)
    model = Word2Vec(vocab_size, embedding_dim, learning_rate, window_size)

    tokens = [word_to_index[word] for word in text]
    pairs = generate_pairs(tokens, window_size)
    for epoch in range(epochs):
        total_loss = 0
        for center, context in pairs:
            negative_samples = sample_negative_samples(vocab_size, center, num_negatives)
            loss = model.train_pair(center, context, negative_samples)
            total_loss += loss

        print(f"epoch: {epoch+1} out of {epochs}, Loss: {total_loss:.4f}")
        
    





