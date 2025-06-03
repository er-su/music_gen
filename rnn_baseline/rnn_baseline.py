import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Dict, Union

class GRUBaseline(nn.Module):
    def __init__(self, hyperparams: Dict[str, Union[int, float]]):
        self.is_cuda = torch.cuda.is_available()
        self.input_len = hyperparams["input_len"]
        self.output_len = hyperparams["output_len"]
        self.lookback = hyperparams["lookback"]

        self.gru_hidden_size = hyperparams["gru_dim"]
        self.num_gru_layers = hyperparams["gru_num_layers"]
        self.dropout_rate = hyperparams["dropout_rate"]

        self.gru_layer = nn.GRU(
            input_size=self.input_len,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_layer,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=False,
        )

        self.ff_dense = nn.Linear(
            in_features=self.gru_hidden_size * self.gru_layer,
            out_features=self.output_len
        )

        self.beat_frac_embd = nn.Embedding(num_embeddings=2048, embedding_dim=64)
    
    def forward(self, z: torch.Tensor):
        pass