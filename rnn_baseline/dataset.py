import sys
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
from torch.utils.data import Dataset, DataLoader

# Add the path to the preprocessor
sys.path.append("../")

from preprocess import Preprocessor, output_type_literal

# Written by Erick

class MidiDataset(Dataset):
    '''
    A wrapper class for the preprocessor for use with Pytorch's built in dataset object.
    If a dataframe path is provided, all other inputs except slice are ignored and the dataframe is loaded and parsed
    '''
    def __init__(
        self, 
        dataframe_path: Union[Path, str] = None,
        sliced: Union[int, None] = None,
        surname: str = None,
        folder_path: Union[Path, str] = Path("surname_checked_midis"),
        get_dict:bool=True,
        binarize:bool=True,
        lookback:int=1,
        resolution:int=8,
        output_type:output_type_literal = "chordify_string",
        collect:bool = False
    ):
        self.dataframe_path = dataframe_path
        self.sliced = sliced
        if self.dataframe_path == None:
            self.preprocessor = Preprocessor(
                folder_path=folder_path,
                get_dict=get_dict,
                binarize=binarize,
                lookback=lookback,
                resolution=resolution,
                output_type=output_type
            )
            if collect:
                self.preprocessor.collect(surname=surname)
                self.data = []
                self.labels = []
                self.durations = []
            
                for i in range(len(self.preprocessor)):
                    dat, label, dur = self.preprocessor[i]
                    self.data.append(dat)
                    self.labels.append(label)
                    self.durations.append(dur)

        else:
            self.dataframe_path = dataframe_path if type(dataframe_path) == Path else Path(dataframe_path)
            self.data, self.labels, self.durations = self.df_to_array(pd.read_csv(dataframe_path))

    def df_to_array(self, dataframe: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        '''
        Converts the loaded dataframe into three separate numpy arrays holding values, labels and durations.
        The dataframe is assumed to have a lookback of 1.
        '''
        data = []
        labels = []
        durations = []
        
        numpy_form = dataframe.to_numpy()
        for i, row in enumerate(numpy_form):
            if row[-1] == 0:
                if i != 0:
                    data.append(np.array(running_data))
                    labels.append(np.array(running_labels))
                    durations.append(np.array(running_durations))

                running_data = []
                running_labels = []
                running_durations = []
                
            running_data.append(row[0])
            running_labels.append(row[1])
            running_durations.append(row[2])

        return (data), (labels), (durations)

    def __len__(self) -> int:
        if self.dataframe_path == None:
            return self.preprocessor.__len__()
        else:
            return len(self.data)
    
    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        If durations are not applicable (i.e. pianoroll or fast_pianoroll), then durations will be completely zero
        '''
        data, labels, durations = self.data[index], self.labels[index], self.durations[index]
        if self.sliced == None:
            return data, labels, durations
        
        if not (len(data) >= self.sliced):
            raise ValueError("Slice length is longer than song")
        
        if durations == None:
            durations = np.zeros(data.shape)

        randint = random.randint(0, len(data) - self.sliced)
        return data[randint: randint + self.sliced], labels[randint: randint + self.sliced], durations[randint: randint + self.sliced]
    
if __name__ == "__main__":
    dataset = MidiDataset(folder_path=Path("../surname_checked_midis"), sliced=32, output_type="fast_pianoroll", surname="Bach")
    loadin = np.load("test.npy", allow_pickle=True)
    for i in range(len(loadin)):
        loadin[i] = Path("../"+loadin[i])
    dataset.preprocessor.filepath_array = loadin

    print(dataset[0])