from mlp import MusicMLP
import pickle as pkl
from pathlib import Path
import torch
from torch import nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torcheval.metrics.functional import binary_f1_score

# Written by Erick
class MLPTrainer():
    def __init__(self, surname):
        data_path=Path(f"../output/{surname}_12note_data.pkl")
        labels_path=Path(f"../output/{surname}_12note_labels.pkl")
        duration_path=Path(f"../output/{surname}_12note_durations.pkl")

        with open(data_path, "rb") as f:
            self.data = pkl.load(f)

        with open(labels_path, "rb") as f:
            self.labels = pkl.load(f)

        with open(duration_path, "rb") as f:
            self.durations = pkl.load(f)
            self.durations = [[round(float(dur), 4) for dur in duration] for duration in self.durations]

        self.model = MusicMLP()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.split_index = round(len(self.data) * 0.90)
        
        pos_weight = torch.tensor([3] * 12)

        self.note_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        self.duration_loss = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

    def train(self, num_epochs, record_accs=False):
        '''
        Perform num_epochs number of train-val cycles\n
        Param: num_epochs - Number of epochs to perform\n
        Param: record_accs - Boolean value of whether the trainer should save train and val F1 score
        '''
        if record_accs:
            self.train_acc = []
            self.val_acc = []
        
        train_data, val_data, train_labels, val_labels, train_durs, val_durs = train_test_split(self.data, self.labels, self.durations, test_size=0.15)

        for epoch in range(num_epochs):
            print(f"###### {epoch} TRAIN #####")

            running_note_loss = 0.0
            running_dur_loss = 0.0
            running_acc = 0.0

            self.model.train()
            for data, labels, durations in zip(train_data, train_labels, train_durs):
                data = torch.tensor(data, dtype=torch.float32).squeeze()
                labels = torch.tensor(labels, dtype=torch.float32)
                durations = torch.tensor(durations, dtype=torch.float32)

                note_preds, duration_preds = self.model(data)
                duration_preds = duration_preds.squeeze()
            
                note_loss = self.note_loss(note_preds, labels)
                duration_loss = self.duration_loss(duration_preds, durations)

                self.optim.zero_grad()
                var_penalty = duration_preds.var(unbiased=False)
                (note_loss + duration_loss - 0.5 * var_penalty).backward()
                self.optim.step()

                running_note_loss += note_loss
                running_dur_loss += duration_loss
                acc = binary_f1_score(note_preds.flatten(), labels.flatten()).item()
                running_acc += acc
            
            if record_accs:
                self.train_acc.append(round(running_acc / len(train_data), 4))
            print(f"Avg note accuracy for epoch {epoch} of {running_acc / len(train_data)}")
            print(f"Note loss for epoch {epoch} of {running_note_loss / len(train_data)}")
            print(f"Duration loss for epoch {epoch} of {running_dur_loss / len(train_data)}")

            print(f"###### {epoch} VALIDATION #####")

            running_note_loss = 0.0
            running_dur_loss = 0.0
            running_acc = 0.0

            self.model.eval()
            for data, labels, durations in zip(val_data, val_labels, val_durs):
                data = torch.tensor(data, dtype=torch.float32).squeeze()
                labels = torch.tensor(labels, dtype=torch.float32)
                durations = torch.tensor(durations, dtype=torch.float32)


                note_preds, duration_preds = self.model(data)
                duration_preds = duration_preds.squeeze()

                note_loss = self.note_loss(note_preds, labels)
                duration_loss = self.duration_loss(duration_preds, durations)

                running_note_loss += note_loss
                running_dur_loss += duration_loss
                acc = binary_f1_score(note_preds.flatten(), labels.flatten()).item()
                running_acc += acc

            if record_accs:
                self.val_acc.append(round(running_acc / len(val_data), 4))
            print(f"Avg note accuracy for epoch {epoch} of {running_acc / len(val_data)}")
            print(f"Note vloss of {running_note_loss / (len(val_data))}")
            print(f"Duration vloss of {running_dur_loss / (len(val_data))}")

    def do_pred(self, seq_len=12):
        '''
        Perform inference from an inital random tensor\n
        Param: seq_len - How many time steps should be generated
        '''
        self.model.eval()
        z = torch.rand(12)
        chords, durations = self.model.infer(z, seq_len=seq_len)
        chords, durations = chords.detach().numpy().squeeze(), durations.detach().numpy().squeeze()

        # Convert durations to string representation of fractions (resolution 12 was used)
        durations = [f"{round(duration * 12)}/12" for duration in durations]
        return chords, durations

    def save_preds(self, chords, durations):
        '''
        Save a set of predictions in an .npy form\n
        Param: chords - Chords to save
        '''
        obj = [chords, durations]
        with open(Path("preds.pkl"), "wb") as f:
            pkl.dump(obj, f)
    
    def save_model(self, save_path=Path("models/model.pt")):
        '''
        Save the model\n
        Param: save_path - Path to where to save the .pt model
        '''
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path=Path("models/model.pt")):
        '''
        Load the model\n
        Param: load_path - Path to where to load the .pt model from
        '''
        self.model.load_state_dict(torch.load(load_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load in the 12note pickle files and train the mlp"
    )
    parser.add_argument(
        "-s", "--surname", type=str, default=None,
        help="If set, search for .pkl file with the given surname"
    )
    args = parser.parse_args()
    surname = args.surname.lower().capitalize()
    train = MLPTrainer(surname)
    train.train(50, record_accs=True)
    chords, durs = train.do_pred(128)
    train.save_preds(chords, durs)



        

