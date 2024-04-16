import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd 

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup

def preprocess_text(text):
    """
    Given an input text, returns the text as a list of preprocessed tokens
    """

    text = text.lower()
    return word_tokenize(text)


def main():

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = 'data\dontpatronizeme_v1.5\DPM_trainingset\dontpatronizeme_pcl.tsv'
    df = pd.read_csv(data, sep='\t', header=None, names=['par_id', 'art_id', 'keyword', 'country_code', 'text','label'])

    # create a toy dataset of the first 100 lines

    toy_dataset = df['text'][:100].tolist()
    toy_labels = df['label'][:100].tolist()

    # convert labels to binary for task 1
    # according to PCL paper, 0 & 1 are considered 'not PCL' (0), while 2-4 are considered 'PCL' (1)

    toy_labels = [0 if x <= 1 else 1 for x in toy_labels]

    # tokenize entries in dataset 
    # tokenized_toy_dataset = [preprocess_text(text) for text in toy_dataset]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for sent in toy_dataset:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
       


main()