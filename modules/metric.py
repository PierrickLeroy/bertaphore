"""A class to compute the attentional burden metrics of a text."""

import torch
from scipy.linalg import toeplitz
import numpy as np

class AttentionalBurden:
    """Main class"""
    def __init__(self, model, tokenizer, prune_cls_sep=True, normalize_attention=True
                 ,w_forward=1, w_backward=1):
        self.model = model
        self.tokenizer = tokenizer
        self.prune_cls_sep = prune_cls_sep
        self.tokens = None
        self.normalize_attention = normalize_attention
        self.burden = None
        self.w_forward = w_forward
        self.w_backward = w_backward

    def _get_attention(self, text):
        """Get the attention matrix for a text"""
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        self.tokens = self.tokenizer.convert_ids_to_tokens(inputs[0])
        if self.prune_cls_sep:
            self.tokens = self.tokens[1:-1]
        outputs = self.model(inputs)
        attention = outputs[-1]
        t = torch.stack(attention)
        if self.prune_cls_sep:
            t = t[:,0,:,1:-1,1:-1]
        else:
            t = t[:,0]
        if self.normalize_attention:
            t = t*t.sum(dim=3, keepdim=True)**-1
        return t

    def compute_attentionCost(self, t, strategy='mean'):
        """Computes the min cost of the flow of attention across the linear graph of tokens.
        The cost is not normalized by the number of tokens.
        Args:
            t (tensor): dim=[num_layers, num_heads, num_tokens, num_tokens]
        Returns:
            float: cost of the flow to match attentional and linear graph
        """
        # aggregation strategy
        if strategy == 'mean':
            t = t.mean(dim=[0,1])
        elif strategy == 'max':
            t = t.max(dim=[0,1])
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        # compute the cost
        t = t/t.sum()
        t_steps = torch.tensor(toeplitz(torch.arange(len(t))*self.w_backward
                                        ,torch.arange(len(t))*self.w_forward
                                        ))
        t_costs = t*t_steps
        return t_costs.sum().item()

    def compute_burden(self, text, strategy='mean'):
        """Compute the attentional burden for a text"""
        t = self._get_attention(text)
        return self.compute_attentionCost(t, strategy)

def compute_maxAttentionalBurden(w_forward, w_backward, n):
    """
    Args:
        w_forward (float): forward weight
        w_backward (float): backward weight
        n (int): length of the sentence

    Returns:
        float: the maximum attentional burden for a graph of length n with unit total attention
    """
    return np.maximum(w_forward, w_backward)*(n-1)

def compute_uniformAttentionalBurden(w_forward, w_backward, n):
    """
    Args:
        w_forward (float): forward weight
        w_backward (float): backward weight
        n (int): length of the sentence

    Returns:
        float: the uniform attentional burden for a graph of length n with unit total attention
    """
    if n==1:
        return 0
    return 1/6*(w_forward+w_backward)*(n+1)

def compute_maxAttentionalBurdenEqualVertices(w_forward, w_backward, n):
    k_0 = np.ceil((n-1)*w_forward/(w_forward+w_backward))
    return 1/n * (k_0*w_forward*(n-(1+k_0)/2) + 1/2*w_backward*(n-k_0)*(n+k_0-1))

def compute_kHopAttentionalBurden(w_forward, w_backward, n, n_0, verbose=False):
    if n==1:
        return 0
    if n_0>=n:
        return np.NaN
    sum_1, sum_2, sum_3, sum_4 = 0, 0, 0, 0

    for k in range(1, min(n-n_0, n_0+1)):
        if verbose:
            print(f"{k} is in sum_1")
        sum_1 += 1/(n_0+k-1) * (w_forward*n_0*(n_0+1)+w_backward*k*(k-1))/2
    sum_1 = 1/n * sum_1

    if n_0>=(n-1)/2:
        # cancelling sums regime
        for k in range(n-n_0, n_0+2):
            if verbose:
                print(f"{k} is in sum_2")
            sum_2 += 1/(n-1)* (w_forward+w_backward) * k*(k-1) /2
        sum_2 = 1/n * sum_2
    else:
        # complete sums regime
        for k in range(n_0+1, n-n_0+1):
            if verbose:
                print(f"{k} is in sum_3")
            sum_3 += 1/(2*n_0)*(w_forward+w_backward) * n_0*(n_0+1)/2
        sum_3 = 1/n * sum_3

    for k in range(max(n-n_0, n_0+1)+1, n+1):
        if verbose:
            print(f"{k} is in sum_4")
        sum_4 += 1/(n_0+n-k) * (w_backward*n_0*(n_0+1)+w_forward*(n-k)*(n-k+1))/2
    sum_4 = 1/n * sum_4

    total = sum_1 + sum_2 + sum_3 + sum_4

    if verbose:
        print(f"n_0={n_0}, sum_1={sum_1}, sum_2={sum_2}, sum_3={sum_3}, sum_4={sum_4}, total={total}")
    return total
