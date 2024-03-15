import pandas as pd
import joblib
import argparse


def get_parser():
    desc = """
    Program: Cas12_prediction
    Version: 0.1
    Author : Liheng Luo
    Email  : <luoliheng@ibms.pumc.edu.cn>

    Optional Cas12s: ['CasMINI', 'OsCas12f1', 'RhCas12f1']

    Examples:
        1) input file:
                ID,SEQ(4bp pam + 20bp target)
                seq1,TTTAACAGGGGATACACCTCCTCT

           """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=desc
    )

    parser.add_argument(
        "-i",
        "--input_filename",
        type=str,
        help="The name of the input file. MUST be a two-column .csv file",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--Cas12f_name",
        choices=["CasMINI", "OsCas12f1", "RhCas12f1"],
        type=str,
        help="The name of the Cas12f to use for prediction.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_filename",
        type=str,
        default="./output.csv",
        help="The name of the output file. (default='./output.csv')",
    )

    return parser


def one_hot_encode(seq):

    cols = [f"pos_{i+1}_{nuc}" for i in range(len(seq)) for nuc in ["A", "T", "C", "G"]]
    one_hot_df = pd.DataFrame(columns=cols, index=[0]).fillna(0)

    for i, nucleotide in enumerate(seq):
        col_name = f"pos_{i+1}_{nucleotide}"
        one_hot_df.at[0, col_name] = 1

    one_hot_df.columns = [
        f"{prefix}_{i}_{nuc}"
        for prefix, end in [("PAM", 5), ("target", 21)]
        for i in range(1, end)
        for nuc in ["A", "T", "C", "G"]
    ]
    return one_hot_df


def main(input_filename, model_name, output_filename):
    input_df = pd.read_csv(input_filename)

    model = joblib.load(f"./sub2/{model_name}/saved_best_model.pkl")

    predictions = model.predict(
        pd.concat(
            [one_hot_encode(seq) for seq in input_df.iloc[:, 1]], ignore_index=True
        )
    )
    print(
        pd.concat(
            [one_hot_encode(seq) for seq in input_df.iloc[:, 1]], ignore_index=True
        )
    )
    print(predictions)
    input_df[f"{model_name}_efficiency"] = predictions
    input_df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.Cas12f_name not in ["CasMINI", "OsCas12f1", "RhCas12f1"]:
        parser.error("One of CasMINI, OsCas12f1, or RhCas12f1 must be chosen.")
    main(args.input_filename, args.Cas12f_name, args.output_filename)
