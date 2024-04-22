import nltk
from nltk.tokenize import word_tokenize
import pandas as pd 
from sklearn.model_selection import train_test_split

"""
Reformats data into the format compatible with our classifier models for task 1.
"""

def main():

    data = 'data\dontpatronizeme_v1.5\DPM_trainingset\dontpatronizeme_pcl.tsv'
    df = pd.read_csv(data, sep='\t', header=None, names=['par_id', 'art_id', 'keyword', 'country_code', 'text','label'])

    # the training dataset has 10469 entries. leave 10% of these entries as our dev set.
    train, dev = train_test_split(df, test_size=0.1, random_state=1) # using random state for controlled shuffles across calls
    
    train_text = train['text'].tolist()
    train_labels = train['label'].tolist() 

    dev_text = dev['text'].tolist()
    dev_labels = dev['label'].tolist() 

    # convert labels to binary for task 1
    # according to PCL paper, 0 & 1 are considered 'not PCL' (0), while 2-4 are considered 'PCL' (1)
    dev_labels = [0 if x <= 1 else 1 for x in dev_labels]
    train_labels = [0 if x <= 1 else 1 for x in train_labels]

    train_df = pd.DataFrame(
        {
            'text': train_text,
            'target': train_labels
        }
    )

    dev_df = pd.DataFrame(
        {
            'text': dev_text,
            'target': dev_labels
        }
    )

    train_df.to_csv('train_dataset.csv')
    dev_df.to_csv('dev_dataset.csv')

    # create gold standard label file for devset for evaluation
    # with open('dev_gold_labels.txt', 'w') as f:
    #     for label in dev_labels:
    #         f.write(str(label) + '\n')


main()