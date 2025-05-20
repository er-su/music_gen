import contextlib
from pathlib import Path
from typing import Literal, Union
import numpy as np
import music21 as m21
import pypianoroll
import tempfile
import shutil

output_type_literal = Literal["chordify_string","chordify_int", "chordify_roman", "pianoroll", "full_pianoroll"]

@contextlib.contextmanager
def make_temp():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

class Preprocessor():

    def __init__(self, folder_path:Union[Path, str] = Path("surname_checked_midis"),
                 get_dict:bool=True,
                 binarize:bool=True,
                 lookback:int=1,
                 resolution:int=8,
                 output_type:output_type_literal = "chordify_string"):
        '''
        Creates a preprocessor object that allows for easy access of piano rolls \n
        Param: get_dict - Returns a dictionary form similar to that of Markov Chains where the key is the length of lookback \n
        Param: binarize - Returns a binarized form of the piano roll \n
        Param: lookback - How much context should be provided to the next prediction \n
        Param: resolution - How often should the the loader sample within the music between quarter notes \n
        Param: output_type - Select what kind of output is generated from the preprocessor
        '''
        if type(folder_path) == str:
            folder_path = Path(folder_path)
        
        self.folder_path = folder_path
        self.binarize = binarize
        self.get_dict = get_dict
        self.lookback = lookback
        self.resolution = resolution
        self.filepath_array = []
        self.output_type = output_type

    def collect(self, surname:str=None):
        '''
        Grabs all file paths that match the surname provided \n
        Param: surname - returns all midi files whos title contains the surname
        '''
        if not self.folder_path.is_dir():
            raise FileNotFoundError("Specificed folder path does not exist")
        
        files = self.folder_path.glob("**/*.mid")
        if surname is None:
            filepath_array = list(files)
        else:
            filepath_array = [filepath for filepath in files if filepath.match(f"{surname}*.mid")]
            if len(filepath_array) <= 0:
                raise FileNotFoundError("No valid files were found containing the specificed surname. Check your spelling")

        self.filepath_array = self.filter(filepath_array)

    def filter(self, filepath_array: list[Path]) -> list[Path]:
        '''
        This is an internal function that filters out any non-valid songs\n
        Param: filepath_array - A List object of paths of midi files\n
        Returns a filtered list of paths
        '''
        filtered_filepath_array = []
        for path in filepath_array:
            music_data = m21.converter.parse(path)
            # Filter if not major
            if music_data.analyze("key").mode != "major":
                continue

            # Filter if more than one time signature
            if len(music_data.getTimeSignatures()) > 1:
                continue

            filtered_filepath_array.append(path)

        return filtered_filepath_array

    def __len__(self):
        return len(self.filepath_array)
    
    def __iter__(self):
        return self.filepath_array.__iter__()
    
    def iter(self):
        '''
        An alternative generator that automatically converts each item into a sliding window dataset\n
        This is opposed to the __iter__ dunder that only iterates through the file paths of filtered midis
        '''
        for path in self.filepath_array:
            yield self.convert_path_to_dict(path)
    
    def __getitem__(self, index):
        return self.convert_path_to_dict(self.filepath_array[index])
    
    
    def convert_path_to_dict(self, path: Path, transpose:bool=True) -> tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
        '''
        Internal function to help convert piano rolls and m21 streams into sliding window
        representations that can be used as training data. \n
        Param: path - Path object to the midi file to convert \n
        Param: transpose - Bool on whether or not to transpose the piece to C Major; should generally always be true\n
        Returns a tuple containing (data, labels if applicable, duration if applicable)
        '''
        music_data = m21.converter.parse(path)
        key = music_data.analyze("key")

        if transpose and key != m21.key.Key("C"):
            i = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
            num_semitones = i.semitones
            music_data.transpose(i, inPlace=True)

        if self.output_type == "chordify_string":
            data = []
            labels = []
            duration = []
            moving_window = ["START"] * self.lookback
            chord_data = music_data.chordify()
            for chord in chord_data.recurse().getElementsByClass(m21.chord.Chord):
                chord.closedPosition(forceOctave=4, inPlace=True)
                data.append(moving_window.copy())
                labels.append(chord.pitchedCommonName)
                duration.append(chord.duration.quarterLength)

                moving_window.pop(0)
                moving_window.append(chord.pitchedCommonName)
            
            return np.array(data), np.array(labels), np.array(duration)
        
        elif self.output_type == "chordify_int":
            data = []
            labels = []
            duration = []
            moving_window = [0] * self.lookback
            chord_data = music_data.chordify()
            for chord in chord_data.recurse().getElementsByClass(m21.chord.Chord):
                chord.closedPosition(forceOctave=4, inPlace=True)
                data.append(moving_window.copy())
                chord_as_int = self.chord_to_base_n(chord)
                labels.append(chord_as_int)
                duration.append(chord.duration.quarterLength)

                moving_window.pop(0)
                moving_window.append(chord_as_int)

            return np.array(data), np.array(labels), np.array(duration)
        
        elif self.output_type == "chordify_roman":
            data = []
            labels = []
            duration = []
            moving_window = ["START"] * self.lookback
            chord_data = music_data.chordify()
            for chord in chord_data.recurse().getElementsByClass(m21.chord.Chord):
                chord_rn = str(m21.roman.romanNumeralFromChord(chord, m21.key.Key("C")).figure)
                chord.closedPosition(forceOctave=4, inPlace=True)
                data.append(moving_window.copy())
                labels.append(chord_rn)
                duration.append(chord.duration.quarterLength)

                moving_window.pop(0)
                moving_window.append(chord_rn)
            
            return np.array(data), np.array(labels), np.array(duration)

        
        # When piano roll, the lookback is forced to be 1 to decrease complexity
        elif self.output_type == "pianoroll":
            with make_temp() as temp_dir:
                output_path = music_data.write("midi", fp=Path(temp_dir, "temp.mid"))
                roll = pypianoroll.read(output_path).set_resolution(self.resolution)

            if len(roll.tracks) > 1:
                print(f"The song {path.name} has more than one track. Exiting...")
                return
            
            roll = roll.tracks[0]
            if self.binarize:
                roll = roll.binarize()
            
            roll = roll.pianoroll.astype(float)
            inputs = np.vstack([np.zeros(128), roll[:-1]])
            
            return inputs, roll, None
        
        elif self.output_type == "full_pianoroll":
            full_roll = pypianoroll.read(path)
            if len(full_roll.tracks) > 1:
                print(f"The song {path.name} has more than one track. Exiting...")
                return
            
            full_roll = full_roll.transpose(num_semitones).tracks[0]
            if self.binarize:
                full_roll = full_roll.binarize()
            
            full_roll = full_roll.pianoroll.astype(float)
            inputs = np.vstack([np.zeros(128), full_roll[:-1]])

            return inputs, full_roll, None

    def chord_to_base_n(self, chord: tuple[m21.note.Note, ...]):
        base_n_sum = 0
        for note in chord:
            relative_to_c4 = note.pitch.midi - 60
            base_n_sum += 2 ** relative_to_c4

        return base_n_sum