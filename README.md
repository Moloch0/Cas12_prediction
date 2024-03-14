# Cas12f_prediction

## 1. Introduction

Cas12f_prediction is a Python script for predicting Cas12f. It takes a CSV file containing sequence information as input, performs prediction using a specified Cas12 (options include CasMINI, OsCas12f1, RhCas12f1), and outputs the prediction results to a CSV file.

It requires `pandas` amd `joblib` two modules.

```shell
pip install pandas joblib
```

## 2. Argument Details

`-i` or `--input_filename`: The name of the input file. It MUST be a two-column CSV file. This is a required argument.
`-n` or `--Cas12f_name`: The name of the Cas12 to use for prediction. It must be one of ‘CasMINI’, ‘OsCas12f1’, ‘RhCas12f1’. This is a required argument.
`-o` or `--output_filename`: The name of the output file. The default is ‘./output.csv’.

## 3. Usage Example

The input file should be a CSV file with two columns: ID and SEQ. SEQ should be a 24bp sequence, including a 4bp pam and a 20bp target. For example:

```csv
ID,SEQ
seq1,TTTAACAGGGGATACACCTCCTCT
```

The command to run the script is as follows:

```python
python prediction.py -i input.csv -n CasMINI -o output.csv
```
