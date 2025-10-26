import numpy as np
import torch
import torch.nn as nn
from mi_estimators import CLUB
class MIGradEstimator(CLUB):
    def __init__(self, x_dim, y_dim, hidden_size):
        super().__init__(x_dim, y_dim, hidden_size)

    def forward(self, x_samples, y_samples, log_probs):
        '''
        Compute the mutual information upper bound
        Args:
            x_samples: samples of x, shape [nsample, x_dim] (Action)
            y_samples: samples of y, shape [nsample, y_dim] (Next State)
            log_prob: log probability of each sample, shape [nsample,] (log prob of action)
        '''
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 
        return (log_probs * (positive.sum(dim = -1) - negative.sum(dim = -1))).mean()