import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mi_estimators import CLUB, DiscreteCLUB

class MIGradEstimator(CLUB):
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
        return (log_probs * (positive.sum(dim = -1) - negative.sum(dim = -1))).mean(), \
        (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
    
class MIGradEstimatorDiscrete(DiscreteCLUB):
    def forward(self, x_samples, y_samples, log_probs):
        """
        Args:
            x_samples: [N, x_dim] one-hot actions
            y_samples: [N, y_dim] one-hot next states
            log_probs: [N,] log π(a|s)
        """
        # q(y|x) logits
        logits = self.q_y_given_x(x_samples)
        log_q_y_given_x = F.log_softmax(logits, dim=-1)

        # Positive log-prob under true pairs
        positive = (log_q_y_given_x * y_samples).sum(dim=-1)  # [N]

        # Negative (mismatched pairs)
        log_probs_neg = torch.matmul(
            log_q_y_given_x, 
            y_samples.T
        )
        negative = log_probs_neg.mean(dim=-1)  # [N]

        # MI signal
        mi_diff = positive - negative

        # Actor MI loss — gradient through log_probs only
        mi_loss = (log_probs * mi_diff).mean()

        return mi_loss, mi_diff.mean()
