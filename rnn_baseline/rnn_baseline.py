import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Dict, Union

class GRUBaseline(nn.Module):
    def __init__(self, hyperparams: Dict[str, Union[int, float]]):
        self.is_cuda = "cuda" if torch.cuda.is_available() else "cpu"
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
            out_features=self.output_len,
        )

        self.sigmoid = nn.Sigmoid()

        self.beat_frac_embd = nn.Embedding(num_embeddings=2048, embedding_dim=64)
    
    def forward(self, z: torch.Tensor, gt_labels: torch.Tensor):
        '''
        Perform a forward pass specifically for training. Utilizes ground truth labels to perform teacher-forcing based training.\n
        Param: z - The inital starting vector. In general, this should be all zeros or a randomized one-hot encoded vector of length 128\n
        Param: gt_labels - Ground truth labels. This should be a tensor of dimensions (batch_size, seq_len, 128)
        '''
        full_in = z
        final_out = torch.zeros(gt_labels.shape)
        prev = torch.zeros(self.num_gru_layers, gt_labels.shape[0], self.gru_hidden_size)
        for i in range(gt_labels.shape[1]):
            preds, prev = self.gru_layer(full_in, prev)
            preds = self.ff_dense(preds)
            preds = self.sigmoid(preds)
            final_out[:, i] = preds

            full_in = gt_labels[:,i]

        return final_out

    def forward(self, z: torch.Tensor, seq_len: int):
        '''
        Perform a forward pass specifically for inference. Does not utilize ground truth labels. Inference is assumed to not be batched\n
        Param: z - The inital starting vector. In general, this should be all zeros or a randomized one-hot encoded vector of length 128\n
        '''
        full_in = z
        final_out = torch.zeros(seq_len, 128)
        prev = torch.zeros(self.num_gru_layers, self.gru_hidden_size)
        for i in range(seq_len):
            preds, prev = self.gru_layer(full_in, prev)
            preds = self.ff_dense(preds)
            preds = self.sigmoid(preds)
            final_out[:, i] = preds

            full_in = preds

        return final_out