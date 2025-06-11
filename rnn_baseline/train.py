import json
import torch
import pickle
import numpy as np
from torch import nn
from pathlib import Path
from dataset import MidiDataset
from typing import Dict, Union
from rnn_baseline import GRUBaseline
from torch.utils.data import DataLoader, random_split, Dataset
from torcheval.metrics.functional import binary_f1_score, binary_auroc

class Trainer:
    def __init__(self, dataset: Dataset, hyperparams: Dict[str, Union[int, float]], checkpoint_path: Path = Path("checkpoints")):
        self.hyperparams = hyperparams
        self.model = GRUBaseline(self.hyperparams)
        self.dataset = dataset
        self.checkpoint_path = checkpoint_path
        self.learning_rate = self.hyperparams["learning_rate"]
        self.loss = FocalLoss(gamma=3, alpha=0.5)
        self.loss2 = nn.BCELoss(reduction="sum")
        self.eps = hyperparams["eps"]

        self.train_set, self.val_set = random_split(self.dataset, [0.95, 0.05])
        self.train_loader = DataLoader(self.train_set, batch_size=self.hyperparams["batch_size"], shuffle=True)
        self.val_load = DataLoader(self.val_set, batch_size=self.hyperparams["batch_size"], shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.is_cuda = torch.cuda.is_available()

        self.num_epochs = self.hyperparams["num_epochs"]
        self.seq_len = self.hyperparams["seq_len"]

    def train_one_epoch(self, epoch_index):
        '''
        Do a single epoch of training
        Param: epoch_index - The current epoch value
        '''
        print(f"######### EPOCH {epoch_index} TRAINING #########")
        self.model.train()
        running_acc = 0.0
        divisor = 0
        for i, datas in enumerate(self.train_loader):
            data, labels, _ = datas
            data = data.type(torch.float32)
            labels = labels.type(torch.float32)
            if self.is_cuda:
                data = data.to("cuda")
                labels = labels.to("cuda")

            preds = self.model(data, labels)
            self.optimizer.zero_grad()
            back = self.loss2(preds, labels)
            back.backward()
            self.optimizer.step()
            running_acc += binary_auroc(data.flatten(), labels.flatten()).item()
            divisor = i + 1
            
            print(f"Batch acc {i} of {running_acc}")
            print(f"Batch {i} loss: {back}")
            
        if self.record_accs:
            self.train_acc.append(round(running_acc /divisor, 4))


    def val_one_epoch(self, epoch_index):
        '''
        Do a single epoch of validation
        Param: epoch_index - The current epoch value
        '''
        print(f"######### EPOCH {epoch_index} VALIDATION #########")
        self.model.eval()
        running_acc = 0.0
        divisor = 0
        for i, datas in enumerate(self.train_loader):
            data, labels, _ = datas
            data = data.type(torch.float32)
            labels = labels.type(torch.float32)
            if self.is_cuda:
                data = data.to("cuda")
                labels = labels.to("cuda")

            preds = self.model(data, labels)
            back = self.loss2(preds, labels)

            running_acc += binary_auroc(data.flatten(), labels.flatten()).item()
            divisor = i + 1

            print(f"Batch acc {i} of {running_acc}")
            print(f"Val batch {i} loss: {back}")
            
        if self.record_accs:
            self.val_acc.append(round(running_acc / divisor, 4))
        
    def train_val(self, record_accs=False):
        '''
        Perform training and validation epoch_num number of times\n
        Param: record_accs - Boolean value whether or not the trainer should save both validation and training accuracies
        '''
    
        self.record_accs = record_accs
        if self.is_cuda:
            self.model.to("cuda")

        if record_accs:
            self.train_acc = []
            self.val_acc = []

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.val_one_epoch(epoch)
            
    def do_pred(self, seq_len=32):
        '''
        Perform inference from an inital random tensor\n
        Param: seq_len - How many time steps should be generated
        '''
        z = torch.rand(128).unsqueeze(axis=0)
        if self.is_cuda:
            self.model.to("cpu")
        
        preds = self.model.infer(z, seq_len)
        return preds
    
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

    def save_preds(self, chords):
        '''
        Save a set of predictions in an .npy form\n
        Param: chords - Chords to save
        '''
        np.save(Path("preds.npy"), chords.detach().numpy())

class FocalLoss(nn.Module):
    '''
    Focal loss based on the loss outlined in Based on paper "Focal loss for dense object detection" by Lin, T.Y., et al
    '''
    def __init__(self, gamma=0, alpha=None, eps=1E-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_alpha = 1 / alpha
        self.eps = eps

    def forward(self, y_pred, gt):
        return -(gt * self.pos_alpha * (1 - y_pred).pow(self.gamma) * (y_pred + self.eps).log() +
            (1 - gt) * self.alpha * y_pred.pow(self.gamma) * (1 - y_pred + self.eps).log()).mean()

if __name__ == "__main__":
    with open("hyperparams.json", "r") as f:
        hyperparams = json.load(f)

    dataset = MidiDataset(dataframe_path="../output/Agnew_chordify_int_data.csv", sliced=256, output_type="fast_pianoroll", collect=False)
    with open("../output/Bach_fast_pianoroll_data.pkl", "rb") as f:
        datas = pickle.load(f)
    
    with open("../output/Bach_fast_pianoroll_labels.pkl", "rb") as f:
        labels = pickle.load(f)

    dataset.data = datas
    dataset.labels = labels
    dataset.durations = [None] * len(datas)    
    
    trainer = Trainer(dataset, hyperparams)
    #trainer.load_model()
    trainer.train_val(record_accs=True)
    preds = trainer.do_pred(seq_len=128)
    trainer.save_model()
    trainer.save_preds(preds)
    