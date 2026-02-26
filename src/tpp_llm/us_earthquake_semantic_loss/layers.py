"""
TPP-LLM Layers
"""
import math
from typing import Union

import torch
from torch import Tensor
from torch import nn


class TimePositionalEncoding(nn.Module):
    """
    Temporal Positional Encoding from THP
    """

    def __init__(self, embedding_dim: int, dtype=torch.float32, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the temporal encoding

        :param embedding_dim: embedding dimension
        :param dtype: data type for the output
        :param device: device
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        i = torch.arange(0, embedding_dim, 1, dtype=self.dtype, device=self.device)
        div_term = (2 * (i // 2).to(self.dtype) * -(math.log(10000.0) / embedding_dim)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, event_times: Tensor) -> Tensor:
        """
        Compute time positional encoding defined in the THP model

        :param event_times: event times, (seq_len, 1)
        :return: temporal encoding vector, (seq_len, embedding_dim)
        """
        result = (event_times * self.div_term).to(self.dtype).to(self.device)
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result