from pathlib import Path
import numpy as np
import music21 as m21
from typing import List, Literal, Tuple, Union, Any
from numpy.typing import NDArray
from fractions import Fraction
import pandas as pd
import pickle as pkl
import random
import argparse

# Writen by Erick

input_type_literal = Literal["chordify_string","chordify_int", "chordify_roman", "pianoroll", "fast_pianoroll", "12note"]

class Postprocessor():

    def __init__(self, input_type: input_type_literal = "chordify_int"):
        self.input_type = input_type

    def extract(self, path: Union[str, Path]) -> np.ndarray:
        '''
        Run this to extract chords and durations to prepare them for postprocessing\n
        If 12note is selected as the method, the filepath is expected to lead to a .pkl file with the format of List[List[12 long one-hot vecs], List[duration]]\n
        If chordify_int or chordify_roman are recieved, then the filepath is expected to be a csv with the first column being chords/ints and the second column being durations\n
        If fast_pianoroll or pianoroll are selected, then the expected output is a .npy file that contains a numpy array of size (seq_len by 128)
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
            with open(path, "rb") as f:
                data_array = pkl.load(f)

            # Chord is of shape (lookback by 12)
            cho = data_array[0]
            dur = data_array[1]
            for chord, duration in zip(cho,dur):
                chords.append(chord)
                durations.append(Fraction(duration))

        elif self.input_type == "pianoroll" or self.input_type == "fast_pianoroll":
            chords = np.load(path)
            return chords, None
            
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

    def postprocess(self, chords: np.ndarray, durs: np.ndarray = None):
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

        elif self.input_type == "pianoroll" or self.input_type == "fast_pianoroll":
            for chord in chords:
                indices = np.where(chord == 1.0)[0].tolist()
                chord = m21.chord.Chord(indices)
                stream.append(chord)


        return stream
    
    def unchordify(self, stream):
        new_stream = m21.stream.Stream()
        
        for chord in stream:
            stream_list = []
            # Find number of notes
            pitches = chord.pitches
            duration = chord.duration
            pitch_classes = chord.pitchClasses

            if len(pitches) == 0:
                continue

            if len(pitches) == 1:
                new_stream.append(chord)
                continue

            num_splits = random.randint(0, len(pitches) - 1)

            if random.random() > 0.7:
                num_splits = len(pitches) - 1
            
            elif self.has_adj(pitch_classes):
                num_splits = len(pitches) * 2

            if num_splits == 0:
                new_chord = m21.chord.Chord(pitches, duration=duration)
                new_stream.append(new_chord)
                continue
            
            pos_splits = list(set([random.randint(1, len(pitches) - 1) for _ in range(num_splits)]))
            pos_splits.insert(0, 0)
            for i in range(len(pos_splits) - 1):
                new_chord = m21.chord.Chord(pitches[pos_splits[i]: pos_splits[i+1]], duration=duration)
                stream_list.append(new_chord)
            
            new_chord = m21.chord.Chord(pitches[pos_splits[-1]: ])
            stream_list.append(new_chord)

            if random.random() >= 0.5:
                stream_list.reverse()

            if len(pitches) <= 3 and random.random() > 0.7:
                random.shuffle(stream_list)

            for chord in stream_list:
                new_stream.append(chord)

        new_stream.insert(0, m21.tempo.MetronomeMark(number=85))
        return new_stream
        # Work with only 3 note chords first with 3 possible variations

    def has_adj(self, arr):
        for i in range(len(arr) - 1):
            if arr[i+1] - arr[i] == 1 or arr[i+1] - arr[i] == 2:
                return True
            
        return False

    def export(self, stream: m21.stream.Stream, path=Path("example_midi/out.mid"), format="midi"):
        if format == "musicxml":
            path = path.with_suffix(".mxl")
            
        stream.write(fmt=format, fp=path)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the MIDI Preprocessor over a folder or single file and export CSV."
    )
    parser.add_argument(
        "-f", "--filepath", type=Path,
        default=Path("mlp/preds.npy"),
        help="Pred file to load in"
    )
    parser.add_argument(
        "--output-type", choices=["chordify_int", "chordify_roman", "12note"],
        default="12note", help="Format of processed preds"
    )
    parser.add_argument(
        "--unchordify", type=bool, default=1,
        help="Whether or not to unchordify"
    )
    parser.add_argument(
        "--format", choices=["musicxml", "midi"], default="midi",
        help="The output format"
    )

    args = parser.parse_args()

    post = Postprocessor(args.output_type)
    chords, durs = post.extract(args.filepath)
    stream = post.postprocess(chords, durs)
    if args.unchordify:
        new_stream = post.unchordify(stream)
        new_stream.insert(0, m21.tempo.MetronomeMark(number=85))
        post.export(new_stream, path=Path("output.mid"), format=args.format)

    else:
        stream.insert(0, m21.tempo.MetronomeMark(number=85))
        post.export(stream, path=Path("output.mid"), format=args.format)