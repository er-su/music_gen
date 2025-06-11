# music_gen

This repository is a holding of a project where we attempt to utilize various machine learning techniques to generate classical music.

## Table of Contents

- [Installation/Dependencies](##installation)
- [Usage](##usage)
- [Examples](##examples)

## Installation

**Clone the repository**
git clone https://github.com/er-su/music_gen.git

**Navigate into the project directory**
cd music_gen

**Install dependencies**
pip install -r requirements.txt

**Download the dataset**
download from the ByteDance [github](https://github.com/bytedance/GiantMIDI-Piano)

## Usage
### Data Preparation

Convert MIDI files into a dataframe for training:

```bash
python df.py -l 5 --output-type <chordify_int|chordify_roman|l2note> --input-dir <surname>
```

- `-l 5`: Sequence length parameter.
- `--output-type`: Format for representation (chordify_roman for svm and decision tree, chordify_int for XGBoost).
- `--s`: Surname of desired composer, if left blank assumed to be all.

### Model Fitting and Generation
- For SVM, enter the svm folder and run either notebook. Assuming the proper data resides in the output folder, a sequence of music should be generated. Choose single for a standard SVM implementation, and batch for the RBFSVM method as described by our paper.
- For Tree, enter the decision_tree folder and run the notebook. A sequence of music should be generated.

## Examples
**MLP**

https://github.com/user-attachments/assets/c360c239-67cb-4b63-96ed-73442a0d9c74

**SVM**

https://github.com/user-attachments/assets/73a76854-540e-4b6f-bbd7-4d0ef91522f6

**Decision Tree**

https://github.com/user-attachments/assets/0db69a33-6067-4ad1-aa19-e9e947c3aa93

**XGBoost**

https://github.com/user-attachments/assets/e212811f-5ce4-4869-9dd5-879fff970966





