import pandas as pd
import sys 

if __name__ == "__main__":
    csv_filename = sys.argv
    df = pd.read_csv(csv_filename)
    df["target"].value_counts