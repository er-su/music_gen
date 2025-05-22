import argparse
from pathlib import Path
import pandas as pd
from preprocess import Preprocessor
import argparse
from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f"[{name}] {t1-t0:.3f}s")

def extract(prep: Preprocessor, artist: str) -> pd.DataFrame:
    """
    Run processing for all collected files and return a DataFrame of chord events.
    """
    print(f"Collecting files for artist '{artist}'...")
    prep.collect(surname=artist)
    print(f"Collected {len(prep.filepath_array)} files.")

    rows = []
    with timer("all-files loop"):
        for path in prep.filepath_array:
            data, labels, durations = prep.convert_path_to_dict(path)
            with timer(f"  file {path.name}"):
                for idx, (inp_ctx, lbl, dur) in enumerate(zip(data, labels, durations)):
                    row = {
                        **{f"lookback_{i+1}": inp_ctx[i] for i in range(prep.lookback)},
                        "label": lbl,
                        "duration": dur,
                        "file": path.name,
                        "position": idx,
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    cols = [f"lookback_{i+1}" for i in range(prep.lookback)] + ["label", "duration", "file", "position"]
    return df[cols]


def main():
    parser = argparse.ArgumentParser(
        description="Run the MIDI Preprocessor over a folder or single file and export CSV."
    )
    # Preprocessor __init__ args
    parser.add_argument(
        "-i", "--folder-path", type=Path,
        default=Path("surname_checked_midis"),
        help="Directory containing .mid files"
    )
    parser.add_argument(
        "--get-dict", dest="get_dict", action="store_true",
        help="Build a Markov-style dict"
    )
    parser.add_argument(
        "--no-get-dict", dest="get_dict", action="store_false",
        help="Do not build a Markov-style dict"
    )
    parser.set_defaults(get_dict=True)
    parser.add_argument(
        "-b", "--binarize", dest="binarize", action="store_true",
        help="Threshold the piano-roll (only for pianoroll output)"
    )
    parser.add_argument(
        "--no-binarize", dest="binarize", action="store_false",
        help="Do not threshold the piano-roll"
    )
    parser.set_defaults(binarize=True)
    parser.add_argument(
        "-l", "--lookback", type=int, default=1,
        help="Context window size for chord sequences"
    )
    parser.add_argument(
        "-r", "--resolution", type=int, default=8,
        help="Sub-divisions per quarter-note (only for pianoroll)"
    )
    parser.add_argument(
        "--output-type", choices=["chordify_string", "chordify_int", "pianoroll"],
        default="chordify_string", help="Format of processed output"
    )
    # collect() args
    parser.add_argument(
        "-s", "--surname", type=str, default=None,
        help="If set, only process MIDI files starting with this string"
    )
    # single-file mode args
    parser.add_argument(
        "--midi-path", type=Path, default=None,
        help="If set, process only this single .mid file"
    )
    parser.add_argument(
        "--no-transpose", dest="transpose", action="store_false",
        help="Do NOT transpose to C major before processing"
    )
    parser.set_defaults(transpose=True)
    # output directory
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("output"),
        help="Directory to save CSV files (default: ./output)"
    )
    args = parser.parse_args()

    print("Instantiating Preprocessor…")
    prep = Preprocessor(
        folder_path=args.folder_path,
        get_dict=args.get_dict,
        binarize=args.binarize,
        lookback=args.lookback,
        resolution=args.resolution,
        output_type=args.output_type
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.midi_path:
        # Single-file CSV
        artist_name = args.surname or args.midi_path.stem
        output_method = args.output_type
        df = extract(prep, artist=artist_name)
        out_file = args.output_dir / f"{artist_name}_{output_method}_data.csv"
        df.to_csv(out_file, index=False)
        print(f"Saved dataframe to {out_file}")
    else:
        # Folder of files
        artist_name = args.surname or "all"
        output_method = args.output_type
        df = extract(prep, artist=artist_name)
        out_file = args.output_dir / f"{artist_name}_{output_method}_data.csv"
        df.to_csv(out_file, index=False)
        print(f"Saved dataframe to {out_file}")

if __name__ == "__main__":
    main()