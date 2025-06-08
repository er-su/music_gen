import json
import torch
from torch import nn
from pathlib import Path
from dataset import MidiDataset
from typing import Dict, Union
from rnn_baseline import GRUBaseline
from torch.utils.data import DataLoader, random_split, Dataset

class Trainer:
    def __init__(self, dataset: Dataset, hyperparams: Dict[str, Union[int, float]], checkpoint_path: Path = Path("checkpoints")):
        
        self.hyperparams = hyperparams
        self.model = GRUBaseline(self.hyperparams)
        self.dataset = dataset
        self.checkpoint_path = checkpoint_path
        self.learning_rate = self.hyperparams["learning_rate"]
        self.loss = nn.BCELoss()

        self.train_set, self.val_set = random_split(self.dataset, [0.95, 0.05])
        self.train_loader = DataLoader(self.train_set, batch_size=self.hyperparams["batch_size"], shuffle=True)
        self.val_load = DataLoader(self.val_set, batch_size=self.hyperparams["batch_size"], shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.is_cuda = torch.cuda.is_available()

        self.num_epochs = self.hyperparams["num_epochs"]
        self.seq_len = self.hyperparams["seq_len"]

    def train_one_epoch(self, epoch_index):
        print(f"######### EPOCH {epoch_index} TRAINING #########")
        self.model.train()
        for i, datas in enumerate(self.train_loader):
            data, labels, _ = datas
            if self.is_cuda:
                data = data.to("cuda")
                labels = labels.to("cuda")

            preds = self.model(data[0], labels)
            back = self.loss(preds, labels)
            back.backward()
            self.optimizer.step()

            print(f"Batch {i} loss: {back}")


    def train_one_epoch(self, epoch_index):
        print(f"######### EPOCH {epoch_index} TRAINING #########")
        self.model.eval()
        for i, datas in enumerate(self.train_loader):
            data, labels, _ = datas
            if self.is_cuda:
                data = data.to("cuda")
                labels = labels.to("cuda")

            preds = self.model(data[0], labels)
            back = self.loss(preds, labels)

            print(f"Val batch {i} loss: {back}")

    def train_val(self):
        if self.cuda:
            self.model.to("cuda")

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            avg_vloss = round(self.val_one_epoch(epoch), 4)

            if epoch % 20 == 0:
                save_fn = f"model_{epoch}_{avg_vloss}.pt"
                torch.save(self.model.state_dict(), self.checkpoint_path / save_fn)

if __name__ == "__main__":
    with open("hyperparams.json", "r") as f:
        hyperparams = json.load(f)
    dataset = MidiDataset(sliced=256, surname="agnew", folder_path=Path("../surname_checked_midis"), output_type="fast_pianoroll")
    trainer = Trainer(dataset, hyperparams)
    trainer.train_val()

