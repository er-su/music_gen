from mlp import MusicMLP
import pickle as pkl
from pathlib import Path
import torch
from torch import nn
from sklearn.model_selection import train_test_split

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
        
        pos_weight = torch.tensor([3] * 12)


        self.note_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        self.duration_loss = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()


    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_data, val_data, train_labels, val_labels, train_durs, val_durs = train_test_split(self.data, self.labels, self.durations, test_size=0.15)
            print(f"###### {epoch} TRAIN #####")

            running_note_loss = 0.0
            running_dur_loss = 0.0

            self.model.train()
            #for i in range(0, self.split_index):
                #data, labels, durations = torch.tensor(self.data[i], dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.float32), torch.tensor(self.durations[i], dtype=torch.float32)
            for data, labels, durations in zip(train_data, train_labels, train_durs):
                data = torch.tensor(data, dtype=torch.float32).squeeze()
                labels = torch.tensor(labels, dtype=torch.float32)
                durations = torch.tensor(durations, dtype=torch.float32)
                note_preds, duration_preds = self.model(data)
                duration_preds = duration_preds.squeeze()
                print(duration_preds.shape)

                note_loss = self.note_loss(note_preds, labels)
                duration_loss = self.duration_loss(duration_preds, durations)

                self.optim.zero_grad()
                (note_loss + duration_loss).backward()
                self.optim.step()

                running_note_loss += note_loss
                running_dur_loss += duration_loss

            print(f"Note loss of {running_note_loss / len(train_data)}")
            print(f"Duration loss of {running_dur_loss / len(train_data)}")

            print(f"###### {epoch} VALIDATION #####")

            running_note_loss = 0.0
            running_dur_loss = 0.0

            self.model.eval()
            #for i in range(self.split_index, len(self.data)):
                #data, labels, durations = torch.tensor(self.data[i], dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.float32), torch.tensor(self.durations[i], dtype=torch.float32)
            for data, labels, durations in zip(val_data, val_labels, val_durs):
                data = torch.tensor(data, dtype=torch.float32).squeeze()
                labels = torch.tensor(labels, dtype=torch.float32)
                durations = torch.tensor(durations, dtype=torch.float32)
                data = data.squeeze()

                note_preds, duration_preds = self.model(data)
                duration_preds = duration_preds.squeeze()

                note_loss = self.note_loss(note_preds, labels)
                duration_loss = self.duration_loss(duration_preds, durations)

                running_note_loss += note_loss
                running_dur_loss += duration_loss

            print(f"Note vloss of {running_note_loss / (len(val_data))}")
            print(f"Duration vloss of {running_dur_loss / (len(val_data))}")

    def do_pred(self):
        self.model.train()
        z = torch.rand(12)
        outs, preds = self.model.infer(z, seq_len=12)
        print(outs)
        print(preds)

if __name__ == "__main__":
    train = MLPTrainer()
    train.train(50)
    train.do_pred()