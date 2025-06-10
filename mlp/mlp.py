import torch
from torch import nn

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
            #CAHNGED
            #nn.Linear(in_features=12 + 1, out_features=256),
            nn.Linear(in_features=12, out_features=64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=1),
            nn.Softplus()
        )
        

    def forward(self, data, durations):
        # Data is of shape seq_len, 12 and should have a set of zeros as its starting point appended
        data_preds = self.note_model(data)
        #CHANGED
        #duration_preds = self.duration_model(torch.cat((data, durations.unsqueeze(axis=-1)), -1))
        duration_preds = self.duration_model(data)

        return data_preds, duration_preds
    
    def infer(self, z, seq_len):
        # Z should be a 12 dim vector
        alpha = 0.75
        temp = 2
        input = z
        notes = torch.zeros((seq_len, 12))
        durations = torch.zeros((seq_len, 1))
        duration_pred = torch.tensor([0.0])

        for i in range(seq_len):
            note_pred = self.note_model(input)
            #CHAGNED
            #duration_pred = self.duration_model(torch.cat((input, duration_pred), -1))
            duration_pred = self.duration_model(input)

            note_pred = torch.sigmoid(note_pred)

            # New stuff
            """ total_mass = note_pred.sum().item()
            p_dyn = total_mass * alpha
            sorted_probs, sorted_idx = torch.sort(note_pred, descending=True)
            cumsum = 0.0
            keep_idx = []

            for i,v in enumerate(sorted_probs):
                cumsum += v.item()
                keep_idx.append(sorted_idx[i].item())

                if cumsum >= p_dyn and len(keep_idx) >= 1:
                    break

            mask = torch.zeros_like(note_pred)
            mask[keep_idx] = 1.0
            filtered = note_pred * mask
            sampled = torch.bernoulli(filtered)
            note_pred = sampled """

            # new stuff again
            """ topk = torch.topk(note_pred, 5)
            mask = torch.zeros_like(note_pred)
            mask.scatter_(0, topk.indices, 1.0) """
            #newer stuff
            total_mass = note_pred.sum().item()
            p_dyn = total_mass * alpha
            topk = torch.topk(note_pred, 12)
            topk_vals = topk.values
            topk_ind = topk.indices
            
            cumsum = 0.0
            valid_inds = []
            for vals, inds in zip(topk_vals, topk_ind):
                cumsum += vals.item()
                valid_inds.append(inds.item())

                if cumsum >= p_dyn:
                    break

            mask = torch.ones_like(note_pred)
            mask = mask * 0.75
            mask[valid_inds] = 1.0 

            
            note_pred = torch.bernoulli(note_pred * mask)

            notes[i] = note_pred
            durations[i] = duration_pred

            input = note_pred
        
        return notes, durations