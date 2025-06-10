from mlp import MusicMLP
import pickle as pkl
from pathlib import Path
import torch
from torch import nn

class MLPTrainer():
    def __init__(self,
                 data_path=Path("../output/Bach_12note_data.pkl"),
                 labels_path=Path("../output/Bach_12note_labels.pkl"),
                 duration_path=Path("../output/Bach_12note_durations.pkl"),
    ):
        
        with open(data_path, "rb") as f:
            self.data = pkl.load(f)

        with open(labels_path, "rb") as f:
            self.labels = pkl.load(f)

        with open(duration_path, "rb") as f:
            self.durations = pkl.load(f)
            self.durations = [[round(float(dur), 4) for dur in duration] for duration in self.durations]

        self.model = MusicMLP()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.split_index = round(len(self.data) * 0.90)
        self.note_loss = nn.BCEWithLogitsLoss()
        self.duration_loss = nn.MSELoss()


    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"###### {epoch} TRAIN #####")

            self.model.train()
            for i in range(0, self.split_index):
                data, labels, durations = torch.tensor(self.data[i], dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.float32), torch.tensor(self.durations[i], dtype=torch.float32)
                data = data.squeeze()
                labels = labels.squeeze()
                durations = durations.squeeze()
                note_preds, duration_preds = self.model(data)

                note_loss = self.note_loss(note_preds, labels)
                duration_loss = self.duration_loss(duration_preds, durations)

                (note_loss + duration_loss).backward()
                self.optim.step()
                print(f"Note loss of {note_loss}")
                print(f"Duration loss of {duration_loss}")

            print(f"###### {epoch} VALIDATION #####")
            self.model.eval()
            for i in range(self.split_index, len(self.data)):
                data, labels, durations = torch.tensor(self.data[i], dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.float32), torch.tensor(self.durations[i], dtype=torch.float32)
                data = data.squeeze()
                labels = labels.squeeze()
                durations = durations.squeeze()
                note_preds, duration_preds = self.model(data)

                note_loss = self.note_loss(note_preds, labels)
                duration_loss = self.duration_loss(duration_preds, durations)

                print(f"Note vloss of {note_loss}")
                print(f"Duration vloss of {duration_loss}")

if __name__ == "__main__":
    train = MLPTrainer()
    train.train(10)



        

