import torch
from torch import nn

# Written by Erick
class MusicMLP(nn.Module):
    def __init__(self):
        super(MusicMLP, self).__init__()
        self.note_model = nn.Sequential(
            nn.Linear(in_features=12, out_features=64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=12),
        )

        self.duration_model = nn.Sequential(
            nn.Linear(in_features=12, out_features=64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=1),
            nn.Softplus()
        )
        

    def forward(self, data):
        '''
        Perform a single forward pass\n
        Param: data - Data is of shape seq_len, 12 and should have a set of zeros as its starting point appended
        '''
        data_preds = self.note_model(data)
        duration_preds = self.duration_model(data)

        return data_preds, duration_preds
    
    def infer(self, z, seq_len):
        '''
        Perform generation\n
        Param: z - Inital vector; should be of length 12
        Param: seq_len - Number of time steps to generate
        '''
        # Z should be a 12 dim vector
        alpha = 0.10
        input = z
        notes = torch.zeros((seq_len, 12))
        durations = torch.zeros((seq_len, 1))
        duration_pred = torch.tensor([0.0])

        for i in range(seq_len):
            note_pred = self.note_model(input)

            duration_pred = self.duration_model(input)

            note_pred = torch.sigmoid(note_pred)

            # Perform variation of top-p selection
            total_mass = note_pred.sum().item()
            print(note_pred.shape)
            p_dyn = total_mass * alpha
            topk = torch.topk(note_pred, 12)
            topk_vals = topk.values
            topk_ind = topk.indices
            
            cumsum = 0.0
            valid_inds = []
            for vals, inds in zip(topk_vals, topk_ind):
                cumsum += vals.item()
                valid_inds.append(inds.item())

                if cumsum >= p_dyn and len(valid_inds) > 0:
                    break

            mask = torch.ones_like(note_pred)
            mask = mask * 0.30
            mask[valid_inds] = 1.0 

            
            note_pred = torch.bernoulli(note_pred * mask)

            notes[i] = note_pred
            durations[i] = duration_pred

            input = note_pred
        
        return notes, durations