#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import requests
import os

API_KEY = None # Set locally

def translate_text(text, target_language="en"):
    """Translate text using the Google Cloud Translation API."""
    api_key = API_KEY
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        'key': api_key,
        'q': text,
        'target': target_language,
        'format': 'text'
    }
    response = requests.post(url, params=params)
    if response.status_code == 200:
        return response.json()['data']['translations'][0]['translatedText']
    else:
        return "Error: " + response.text

def translate_csv(input_file, output_file, text_column, target_language="en"):
    """Translate a column in a CSV file and write to a new file."""
    df = pd.read_csv(input_file)
    df[text_column] = df[text_column].apply(lambda text: translate_text(text, target_language))
    df.to_csv(output_file, index=False)
    print(f"Translated CSV saved as {output_file}")


# In[11]:


input_csv = 'CCPC_Dataset.csv'  # Path to your source CSV
output_csv = 'translated_CCPC.csv'  # Path for the translated CSV
column_to_translate = 'text'  # Column name in your CSV that contains the text
target_lang = 'en'  

translate_csv(input_csv, output_csv, column_to_translate, target_lang)


# In[ ]:


model = torch.load('bert_model_EngWithStopwords') # load the model trained on English data. 
# Note that the entire model is too large to push to GitHub, so make sure to run the English data training script first, which will save the model


# In[ ]:


df_test = pd.read_csv('translated_CCPC.csv')

df_test['text'] = df_test['text'].apply(lambda x:clean_text(x))
test_sentences = df_test['text'].values


# In[ ]:


# for i in df_test['text']:
#     try: clean_text(i) 
#     except AttributeError:
#         print(i)

df_test['text'] = df_test['text'].apply(lambda x:clean_text(x))
test_sentences = df_test['text'].values


# In[ ]:


test_dataset = TensorDataset(test_input_ids, test_attention_masks)
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# In[ ]:


predictions = []
for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        with torch.no_grad():
            output= model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask)
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()

            predictions.extend(list(pred_flat))


# In[ ]:


df_output = pd.DataFrame()
#df_output['id'] = df_test['id'] # Do we need to add ids?
df_output['target'] =predictions
results_filename = 'D4_OriginalModelWithTranslation.out'

df_output.to_csv(results_filename,index=False, header=False)


# In[ ]:


# EVALUATION

import sys
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd

test_set_csv = 'translated_CCPC.csv'

# read in the CSV and get the gold labels
gold_df = pd.read_csv(test_set_csv)
task1_gold = gold_df['target'].tolist()

# read in the results file and get the system output labels
task1_res = []
with open(results_filename) as f:
    for line in f:
        task1_res.append(int(line.strip()))

with open('D4_OriginalModelWithTranslation_Results.txt', 'w') as outf:

    # task 1 scores
    t1p = precision_score(task1_gold, task1_res)
    t1r = recall_score(task1_gold, task1_res)
    t1f = f1_score(task1_gold, task1_res)
    # task1
    outf.write('task1_precision:'+str(t1p)+'\n')
    outf.write('task1_recall:'+str(t1r)+'\n')
    outf.write('task1_f1:'+str(t1f)+'\n')    

