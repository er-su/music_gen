from pathlib import Path
import numpy as np
import music21 as m21
from typing import List, Literal, Tuple, Union, Any
from numpy.typing import NDArray
from fractions import Fraction
import pandas as pd
import pickle as pkl

input_type_literal = Literal["chordify_string","chordify_int", "chordify_roman", "pianoroll", "fast_pianoroll", "12note"]

class Postprocessor():

    def __init__(self, input_type: input_type_literal = "chordify_int"):
        self.input_type = input_type

    def extract(self, path: Union[str, Path]) -> np.ndarray:
        '''
        Run this to extract chords and durations to prepare them for postprocessing\n
        If 12note is selected as the method, the filepath is expected to lead to a .pkl file with the format of List[List[12 long one-hot vecs], List[duration]]\n
        If chordify_int or chordify_roman are recieved, then the filepath is expected to be a csv with the first column being chords/ints and the second column being durations
        '''
        if type(path) == str:
            path = Path(path)

        chords = []
        durations = []

        if self.input_type == "chordify_roman":
            data_array = pd.read_csv(path).to_numpy()
            for chord, duration in data_array:
                chords.append(str(chord))
                durations.append(Fraction(duration))

        elif self.input_type == "chordify_int":
            data_array = pd.read_csv(path).to_numpy()
            for chord, duration in data_array:
                chords.append(int(chord))
                durations.append(Fraction(duration))

        elif self.input_type == "12note":
            data_array = pkl.load(path)
            print(data_array)
            # Chord is of shape (lookback by 12)
            for chord, duration in data_array:
                chords.append(chord)
                durations.append(Fraction(duration))
            
        else:
            raise ValueError("Not supported")
        
        return chords, durations

    # Do not use
    def from_file(self, path: Union[str, Path]) -> np.ndarray[Tuple[int, Fraction]]:
        if type(path) == str:
            path = Path(path)

        with open(path, "r") as f:
            lines = f.readlines()

        val: List[Tuple[str, str]] = []
        for line in lines:
            line = line.split("|")
            val.append((int(line[0].strip()), Fraction(line[1].strip())))

        return np.array(val)

    # Do not use
    def from_folder(self, path: Union[str, Path]) -> np.ndarray[np.ndarray[Tuple[int, Fraction]]]:
        if type(path) == str:
            path = Path(path)

        val = []
        for output in path.glob("**/*.txt"): # type: ignore
            val.append(self.from_file(output))

        return np.array(val)
    
    def base_n_to_chord(self, index: int, duration: Fraction = 0.25, num_keys: int = 12) -> m21.chord.Chord:
        string = f"{index:012b}"[::-1]
        combo1 = []
        for pos, val in enumerate(string): 
            if int(val) == 1:
                if pos == 12:
                    combo1.append(0)
                else:
                    combo1.append(pos)

        return m21.chord.Chord(combo1, duration=m21.duration.Duration(duration))

    def postprocess(self, chords: np.ndarray, durs: np.ndarray):
        stream = m21.stream.Stream()
        
        if self.input_type == "chordify_int":
            for chord, dur in zip(chords, durs):
                stream.append(self.base_n_to_chord(chord, dur))

        elif self.input_type == "chordify_roman":
            for roman, dur in zip(chords, durs):
                duration = m21.duration.Duration(dur)
                chord = m21.chord.Chord(m21.roman.RomanNumeral(roman), duration=duration)
                stream.append(chord)

        elif self.input_type == "12note":
            # chords is of shape (seq_len, 12)
            for chord, dur in zip(chords, durs):
                duration = m21.duration.Duration(dur)
                indices = np.where(chord == 1.0)[0].tolist()
                chord = m21.chord.Chord(indices, duration=duration)
                stream.append(chord)

        return stream

if __name__ == "__main__":
    post = Postprocessor("chordify_int")
    chords, durs = post.extract("generated_sequence_batched.csv")
    print(chords)
    stream = post.postprocess(chords, durs)
    stream.show("musicxml")