"""A class to compute the attentional burden metrics of a text."""

import torch
from scipy.linalg import toeplitz

class AttentionalBurden:
    """Main class"""
    def __init__(self, model, tokenizer, prune_cls_sep=True, normalize_attention=True):
        self.model = model
        self.tokenizer = tokenizer
        self.prune_cls_sep = prune_cls_sep
        self.tokens = None
        self.normalize_attention = normalize_attention
        self.burden = None

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

    @staticmethod
    def compute_attentionCost(t, strategy='mean'):
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
        t_steps = torch.tensor(toeplitz(torch.arange(len(t)), torch.arange(len(t))))
        t_costs = t*t_steps
        return t_costs.sum().item()

    def compute_burden(self, text, strategy='mean'):
        """Compute the attentional burden for a text"""
        t = self._get_attention(text)
        return self.compute_attentionCost(t, strategy)
 