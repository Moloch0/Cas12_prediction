# Cas12f_prediction

## 1. Introduction

Cas12f_prediction is a Python script for predicting Cas12f. It takes a CSV file containing sequence information as input, performs prediction using a specified Cas12 (options include CasMINI, OsCas12f1, RhCas12f1, enAsCas12f1, SpaCas12f1), and outputs the prediction results to a CSV file.

```shell
pip install pandas numpy viennarna biopython onnxruntime
git clone https://github.com/Moloch0/Cas12f_prediction.git
cd Cas12f_prediction
wget https://github.com/Moloch0/Cas12f_prediction/releases/download/models/models.tar.gz
tar -xvf models.tar.gz
```

## 2. Argument Details

`-i` or `--input_filename`: The name of the input file. It MUST be a two-column CSV file. This is a required argument.

`-n` or `--Cas12f_name`: The name of the Cas12 to use for prediction. It must be one of 'CasMINI', 'OsCas12f1', 'RhCas12f1', 'enAsCas12f1', 'SpaCas12f1'. This is a required argument.

`-o` or `--output_filename`: The name of the output file. The default is './output.csv'.

## 3. Usage Example

The input file should be a CSV file with two columns: ID and SEQ. SEQ should be a 24bp sequence, including a 4bp pam and a 20bp target. For example:

```csv
ID,SEQ(6bp before + 4bp pam + 20bp target + 6bp after)
seq1,AGCGCTATTACAGCTCGCAGATCTGCACCCGGGAAA
seq2,GCTGATTTTATCTCCACGTGCCCTGAAGGTTAACCT
```

The command to run the script is as follows:

```python
python prediction.py -i example_input.csv -n OsCas12f1 -o example_output.csv
```
