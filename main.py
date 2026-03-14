
from train import train
if __name__ == "__main__":
    
    with open("shakespeare.txt", "r") as f:
        text = f.read().lower().split()

    train(text, embedding_dim=50, learning_rate=0.01, window_size=2, epochs=2)