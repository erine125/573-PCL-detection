import pandas as pd
import sys
from easynmt import EasyNMT

if __name__ == "__main__":
    selected_df = pd.read_csv("selected.csv")

    model = EasyNMT('opus-mt')

    sentences_ch = selected_df["text"].tolist()
    sentences_en = model.translate(sentences_ch, target_lang='en')
    print(sentences_en)