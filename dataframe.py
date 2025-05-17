import pandas as pd
from pathlib import Path
from preprocess import Preprocessor

print("Instantiating Preprocessor…")
prep = Preprocessor(
    folder_path=Path("surname_checked_midis"),
    output_type="chordify_string"
)

artist = "Bach"
print("Collecting files…")
prep.collect(surname=artist)
print(f"Collected {len(prep.filepath_array)} files.")

rows = []
for path in prep.filepath_array:
    data, labels, durations = prep.convert_path_to_dict(path)
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
df = df[cols]

print(df.head())

output_path = Path("output") / f"{artist}_chord_data.csv"
output_path.parent.mkdir(exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved dataframe to {output_path}")
