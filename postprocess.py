from pathlib import Path
import numpy as np
import music21 as m21
from typing import List, Literal, Tuple, Union, Any
from numpy.typing import NDArray

input_type_literal = Literal["chordify_string","chordify_int", "chordify_roman", "pianoroll", "full_pianoroll"]

class Postprocessor():
    '''
    Ouputs are expected to follow the following format: "chordify_int|duration"
    '''
    def __init__(self, input_type: input_type_literal = "chordify_int"):
        self.input_type = input_type
    
    def from_file(self, path: Union[str, Path]) -> np.ndarray[Tuple[int, float]]:
        if type(path) == str:
            path = Path(path)

        with open(path, "r") as f:
            lines = f.readlines()

        val: List[Tuple[str, str]] = []
        for line in lines:
            line = line.split("|")
            val.append((int(line[0].strip()), float(line[1].strip())))

        return np.array(val)

    def from_folder(self, path: Union[str, Path]) -> np.ndarray[np.ndarray[Tuple[int, float]]]:
        if type(path) == str:
            path = Path(path)

        val = []
        for output in path.glob("**/*.txt"): # type: ignore
            val.append(self.from_file(output))

        return np.array(val)
    
    def base_n_to_chord(self, index: int, duration: float = 0.25, num_keys: int = 12) -> m21.chord.Chord:
        string = f"{index:012b}"[::-1]
        combo = [pos for pos, val in enumerate(string) if int(val) == 1]
        return m21.chord.Chord(combo, duration=m21.duration.Duration(duration))

    def postprocess(self, chords: np.ndarray[Tuple[str, str]]):
        stream = m21.stream.Stream()
        for chord in chords:
            stream.append(self.base_n_to_chord(chord[0], chord[1]))

        return stream