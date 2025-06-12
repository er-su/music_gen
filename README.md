# music_gen

This repository is a holding of a project where we attempt to utilize various machine learning techniques to generate classical music.

## Table of Contents

- [Installation/Dependencies](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributions](#contributions)

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/er-su/music_gen.git
```
**2. Navigate into the project directory**
```bash
cd music_gen
```
**3. Install dependencies**
```bash
pip install -r requirements.txt
```
**4. Download the dataset**

Download from the ByteDance [github](https://github.com/bytedance/GiantMIDI-Piano). In this project, we utilized the surname_checked_midis directory

## Usage
### Data Preparation

Convert MIDI files into a dataframe to prepare for training:

```bash
python df.py --lookback 5 --output-type <chordify_int|chordify_roman|l2note> --surname <surname>
```

- `-l 5`: Sequence length parameter.
- `--output-type`: Format for representation (chordify_roman for SVM and decision tree, chordify_int for XGBoost).
- `--s`: Surname of desired composer, if left blank assumed to be all.

For the MLP, since it uilitzes the 12note preprocessing method, special steps are required to train the files.
- Enter the mlp directory
- Run the following command
```bash
python 12note_pickle.py --folder-path <Path> --surname <str> --output-dir <Path>
```
- `--folder-path`: The path to the dataset directory.
- `--surname`: Surname of desired composer, if left blank assumed to be all.
- `--output-dir`: The path to the directory of where to output the .pkl files

Example versions of this already exist under the *output* directory

### Model Fitting and Generation
- For SVM, enter the svm folder and run either notebook. Assuming the proper data resides in the output folder, a sequence of music should be generated. Choose single for a standard SVM implementation, and batch for the RBFSVM method as described by our paper.
- For Decision Tree, enter the decision_tree folder and run the notebook. A sequence of music should be generated.
- For the Multilayer Perceptron, enter the mlp directory and run the mlp.py file. If default arguments were utilized in the 12note_pickle.py step, then no modification is necessary. train.py takes in a positional argument surname that should be exact same as the one used when creating the .pkl files.
```bash
python train.py --surname <str>
```
- For XGBoost, enter the folder xgboost and run the notebook. Assuming the correct dataset exists, this will generate a predicted sequence of 128.

Example versions of this already exist under the *final_out_pred* directory

### Postprocessing
- Depending on the model used, the outputs should be saved in different formats. If XGBoost, SVM, or Decision Trees were used, the output will be a .csv file with a column for chords and a column for durations
- If the MLP was used, then a .pkl file is generated containing both the durations and the predicted chords.
- Run the following command in the main directory
```bash
python postprocess.py --filepath <Path> --output-type <str> --unchordify <bool> --format <str>
```
- `--filepath`: The path to the predictions.
- `--output-type`: Type of preprocessing that was utilized. Options include chordify_roman, chordify_int, and 12note
- `--unchordify`: Whether or not the program should attempt to unchordify the predictions
- `--format`: What format to save the predictions as. Options include .mid for midi files and .mxl for musicxml files

Example midi version already exist under the *example_midi* directory

## Examples
**MLP**

https://github.com/user-attachments/assets/c360c239-67cb-4b63-96ed-73442a0d9c74

**SVM**

https://github.com/user-attachments/assets/73a76854-540e-4b6f-bbd7-4d0ef91522f6

**Decision Tree**

https://github.com/user-attachments/assets/0db69a33-6067-4ad1-aa19-e9e947c3aa93

**XGBoost**

https://github.com/user-attachments/assets/e212811f-5ce4-4869-9dd5-879fff970966

## Contributions
Erick Sun:
- Preprocess
- Postprocess
- MLP
- RNN Baseline
  
Kieran Pazmino 
- Decision Tree
- SVM
- Dataframe

Jayden Malhotra
- XGBoost

Thank you Jayden for continuing to help us despite dropping the class



