from preprocess import Preprocessor
import numpy as np
from pathlib import Path
import pickle as pkl
import argparse

# Written by Erick
# Used to extract 12note preprocessing of a specific surname and save as .pkl files for later use
def extract(pre, artist, out_folder):
    pre.collect(artist)
    datas = []
    labels = []
    durations = []

    for i in range(len(pre)):
        data, label, duration = pre[i]
        datas.append(data)
        labels.append(label)
        durations.append(duration)
    
    with open(out_folder / f"{artist}_12note_data.pkl", "wb") as f:
        pkl.dump(datas, f)
    print(f"Created data file at {out_folder}")
    with open(out_folder / f"{artist}_12note_labels.pkl", "wb") as f:
        pkl.dump(labels, f)
    print(f"Created label file at {out_folder}")
    with open(out_folder / f"{artist}_12note_durations.pkl", "wb") as f:
        pkl.dump(durations, f)   
    print(f"Created duration file at {out_folder}")
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the MIDI Preprocessor over a folder and generate 12note pickle files"
    )
    parser.add_argument(
        "-i", "--folder-path", type=Path,
        default=Path("surname_checked_midis"),
        help="Directory containing .mid files"
    )
    parser.add_argument(
        "-s", "--surname", type=str, default=None,
        help="If set, only process MIDI files starting with this string"
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("output"),
        help="Directory to save .pkl files (default: ./output)"
    )

    args = parser.parse_args()
    pre = Preprocessor(
        folder_path=args.folder_path,
        output_type="12note"
    )

    surname = args.surname.lower().capitalize()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract(pre, surname, args.output_dir)