import contextlib
from pathlib import Path
from typing import Literal
import numpy as np
import music21 as m21
import pypianoroll
import tempfile
import shutil

output_type_literal = Literal["chordify", "pianoroll"]

@contextlib.contextmanager
def make_temp():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

class Preprocessor():

    def __init__(self, folder_path:Path=Path("surname_checked_midis"),
                 get_dict:bool=True,
                 binarize:bool=True,
                 lookback:int=1,
                 resolution:int=8,
                 output_type:output_type_literal = "chordify"):
        '''
        Creates a preprocessor object that allows for easy access of piano rolls \n
        Param: get_dict - Returns a dictionary form similar to that of Markov Chains where the key is the length of lookback \n
        Param: binarize - Returns a binarized form of the piano roll \n
        Param: lookback - How much context should be provided to the next prediction \n
        Param: resolution - How often should the the loader sample within the music between quarter notes
        '''
        self.folder_path = folder_path
        self.binarize = binarize
        self.get_dict = get_dict
        self.lookback = lookback
        self.resolution = resolution
        self.filepath_array = []
        self.output_type = output_type

    def collect(self, surname:str=None):
        '''
        Grabs all file paths that match the surname provided
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

    def filter(self, filepath_array: list[Path]):
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
        for path in self.filepath_array:
            yield self.convert_path_to_dict(path)
    
    def __getitem__(self, index):
        return self.convert_path_to_dict(self.filepath_array[index])
    
    
    def convert_path_to_dict(path: Path,
                             lookback:int=1,
                             binarized:bool=True,
                             transpose:bool=True,
                             resolution:int=8,
                             output_type:output_type_literal="chordify"
    ):
        data = []
        labels = []
        duration = []
        music_data = m21.converter.parse(path)
        key = music_data.analyze("key")

        if transpose and key != m21.key.Key("C"):
            i = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
            music_data.transpose(i, inPlace=True)

        if output_type == "chordify":
            moving_window = ["START"] * lookback
            chord_data = music_data.chordify()
            for chord in chord_data.recurse().getElementsByClass(m21.chord.Chord):
                data.append(moving_window.copy())
                labels.append(chord.pitchedCommonName)
                duration.append(chord.duration.quarterLength)

                moving_window.pop(0)
                moving_window.append(chord.pitchedCommonName)
            
            return np.array(data), np.array(labels), np.array(duration)
        
        # When piano roll, the lookback is forced to be 1 to decrease complexity
        if output_type == "pianoroll":
            with make_temp() as temp_dir:
                output_path = music_data.write("midi", fp=Path(temp_dir / "temp.mid"))
                roll = pypianoroll.read(output_path).set_resolution(resolution)

            if binarized:
                roll = roll.binarize()

            input = np.stack([np.zeros(128), roll[:-1]], axis=0)
            
            return input, roll, None

