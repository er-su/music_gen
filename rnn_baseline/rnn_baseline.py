import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Dict, Union

# Written by Erick
class GRUBaseline(nn.Module):
    def __init__(self, hyperparams: Dict[str, Union[int, float]]):
        super().__init__()
        self.is_cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_len = hyperparams["input_len"]
        self.output_len = hyperparams["output_len"]
        self.lookback = hyperparams["lookback"]
        self.seq_len = hyperparams["seq_len"]

        self.gru_hidden_size = hyperparams["gru_dim"]
        self.num_gru_layers = hyperparams["gru_num_layers"]
        self.dropout_rate = hyperparams["dropout_rate"]

        self.gru_layer = nn.GRU(
            input_size=self.input_len,
            hidden_size=self.gru_hidden_size,
            num_layers=self.num_gru_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=False,
        )

        self.ff_dense = nn.Linear(
            in_features=self.gru_hidden_size,
            out_features=self.output_len,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, data: torch.Tensor, gt_labels: torch.Tensor):
        '''
        Perform a forward pass specifically for training. Utilizes ground truth labels to perform teacher-forcing based training.\n
        Param: data - Ground truth labels shifted by one. In general, the first vector should be all zeros or a randomized one-hot encoded vector of length 128\n
        Param: gt_labels - Ground truth labels. This should be a tensor of dimensions (batch_size, seq_len, 128)
        '''
        preds, _ = self.gru_layer(data)
        preds = self.ff_dense(preds)
        preds = self.sigmoid(preds)
        return preds

    def infer(self, z: torch.Tensor, seq_len: int = 32):
        '''
        Perform a forward pass specifically for inference. Does not utilize ground truth labels. Inference is assumed to not be batched\n
        Param: z - The inital starting vector. In general, this should be all zeros or a randomized one-hot encoded vector of length 128\n
        '''
        full_in = z
        final_out = torch.zeros((seq_len, 128))
        prev = torch.zeros((self.num_gru_layers, self.gru_hidden_size))
        for i in range(seq_len):
            preds, prev = self.gru_layer(full_in, prev)
            preds = self.ff_dense(preds)
            preds = self.sigmoid(preds)

            # Perform a variation of top-p selection
            preds = preds.squeeze()
            print(preds.shape)
            total_mass = preds.sum().item()
            p_dyn = total_mass * 0.3
            topk = torch.topk(preds, 64)
            topk_vals = topk.values
            topk_ind = topk.indices
            cumsum = 0.0
            valid_inds = []
            for vals, inds in zip(topk_vals, topk_ind):
                cumsum += vals.item()
                valid_inds.append(inds.item())

                if cumsum >= p_dyn and len(valid_inds) > 2:
                    break

            mask = torch.ones_like(preds)
            mask = mask * 0.75
            mask[valid_inds] = 1.0 

            preds = torch.bernoulli(preds * mask)
            final_out[i] = preds.squeeze()

            full_in = preds.unsqueeze(dim=0)

        return final_out