import torch
from torch import nn

class MusicMLP(nn.Module):
    def __init__(self):
        super(MusicMLP, self).__init__()
        self.note_model = nn.Sequential(
            nn.Linear(in_features=12, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=12),
        )

        self.duration_model = nn.Sequential(
            nn.Linear(in_features=12, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=1),
            nn.ReLU()
        )

    def forward(self, data):
        # Data is of shape seq_len, 12 and should have a set of zeros as its starting point appended
        data_preds = self.note_model(data)
        duration_preds = self.duration_model(data)

        return data_preds, duration_preds
    
    def infer(self, z, seq_len):
        # Z should be a 12 dim vector
        input = z
        notes = torch.zeros((seq_len, 12))
        durations = torch.zeros((seq_len, 1))

        for i in range(seq_len):
            note_pred = self.note_model(input)
            duration_pred = self.duration_model(input)

            notes[i] = note_pred
            durations[i] = duration_pred

            input = note_pred
        
        return notes, durations

        

    
