# Word2Vec with Skip-Gram Negative Sampling

This project implements the Word2Vec algorithm using the Skip-Gram model with Negative Sampling.

The loss function we wan to minimise:

$$
\min_{\theta} -\log \sigma \left(\theta_{C}[gt]^T\theta_W[i]\right) - \sum_{k=1}^K \log \left[ \sigma\left(-\theta_{C}[k]^T\theta_W[i]\right)\right]
$$

where:
- $\(\theta_W[i]\)$ is the embedding of the target word,
- $\(\theta_{C}[gt]\)$ is the embedding of the ground truth context word,
- $\(\theta_{C}[k]\)$ are the embeddings of the negative samples,
- $\(\sigma\)$ is the sigmoid function.

## Dataset

A toy dataset based on Shakespeare's sonnets is used for training and testing the model.

## Usage

To run the program, execute:

```bash
python3 main.py
```

## Requirements

- numpy
