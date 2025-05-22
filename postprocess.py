from pathlib import Path
import numpy as np
from typing import List, Literal, Tuple, Union

input_type_literal = Literal["chordify_string","chordify_int", "chordify_roman", "pianoroll", "full_pianoroll"]


class Postprocessor():
    def __init__(self, input_type: input_type_literal = "chordify_string"):
        self.input_type = input_type
    
    def from_file(self, path: Union[str | Path]) -> np.ndarray[Tuple[str, str]]:
        if type(path) == str:
            path = Path(path)

        with open(path, "r") as f:
            lines = f.readlines()

        val = []
        for line in lines:
            line = line.split("|")
            val.append((line[0].strip(), line[1].strip()))

        return np.array(val)

    def from_folder(self, path: Union[str | Path]) -> np.ndarray[np.ndarray[Tuple[str, str]]]:
        if type(path) == str:
            path = Path(path)

        val = []
        for output in path.glob("**/*.txt"):
            val.append(self.from_file(output))

        return np.array(val)
        

