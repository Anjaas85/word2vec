import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Word2Vec:
    def __init__(self, voccabulary_size, embedding_dim, learning_rate = 0.001, window_size = 5):
        self.voccabulary_size = voccabulary_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.window_size = window_size

        # init of weights not to 0 to avoid vanishing gradient problem
        self.Win = np.random.rand(voccabulary_size, embedding_dim)  
        self.Wout = np.random.rand(voccabulary_size,embedding_dim)  




    def train_pair(self, center_word, context_word, negative_samples):
        
        vc = self.Win[center_word].T
        vcont = self.Wout[context_word] 

        #loss = - log(sigm(poss)) -  sum of log(sigm(-neg))
        # loss = -log(sigm(vc * vcont)) - sum(log(sigm(-vc * vneg))) 

        # positives 
        #sigm(z) = 1 - sigm(-z)
        sigmoid_pos = sigmoid(np.dot(vc, vcont))

        loss = -np.log(sigmoid_pos)

        grad_pos = self.learning_rate * (1 - sigmoid_pos)
        
        self.Win[center_word] += grad_pos * vcont  # we add because we want to point in the direction of the gradient to minimize the loss
        self.Wout[context_word] += grad_pos * vc 

        # negatives
        for neg in negative_samples:
            vneg = self.Wout[neg]
            sigmoid_neg = sigmoid(-np.dot(vc, vneg))
            
            loss -= np.log(sigmoid_neg)
            
            grad_neg = self.learning_rate * (1 - sigmoid_neg)
            
            
            self.Win[center_word] -= grad_neg * vneg    # we subtract because we want to minimize the loss
            self.Wout[neg] -= grad_neg * vc
        return loss








