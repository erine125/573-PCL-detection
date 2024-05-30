#!/usr/bin/env python
# coding: utf-8

# In[64]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
import pandas as pd

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup


# In[65]:


import numpy as np
import time
import datetime
import random
from nltk.corpus import stopwords
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from nltk.corpus import wordnet


# In[66]:


REMOVE_STOPWORDS = False


# In[67]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# In[68]:


## to run it on Ca

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[69]:


# Transform to pd dataframe and print first rows
df = pd.read_csv("data\\split_data\\train_dataset.csv")
df.head(n=10)


# In[70]:


# This is the preprocessing from the tutorial. We may adapt it however we want.
# This one is VERY strict so it removes stopwords, emojis, punctuation, etc.

sw = stopwords.words('english')

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    text = re.sub(r"http\S+", "",text) #Removing URLs
    #text = re.sub(r"http", "",text)

    html=re.compile(r'<.*?>')

    text = html.sub(r'',text) #Removing html tags

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') #Removing punctuations

    if REMOVE_STOPWORDS:

        text = [word.lower() for word in text.split() if word.lower() not in sw]

        text = " ".join(text) #removing stopwords

    else:
        text = [word.lower() for word in text.split()]

        text = " ".join(text) #removing stopwords


    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) #Removing emojis

    return text


# In[71]:


# Applied preprocessing


# for index, row in df.iterrows():
#     if type(row['text']) == float:
#         print(row['text'])
#         print(index)

df['text'] = df['text'].apply(lambda x: clean_text(x))
df.head()
len(df)


# In[72]:


sentences = df.text.values
labels = df.target.values
random_seed = 42 


# In[73]:


def split_data_randomly(sentences, labels, train_size, random_seed=None):
    """
    Split the data randomly into training and validation sets.

    Parameters:
    - sentences: numpy array containing sentences
    - labels: numpy array containing corresponding labels
    - train_size: proportion of the dataset to include in the training set
    - random_seed: seed for random number generator (optional)

    Returns:
    - train_sentences: numpy array containing training sentences
    - train_labels: numpy array containing corresponding training labels
    - val_sentences: numpy array containing validation sentences
    - val_labels: numpy array containing corresponding validation labels
    """

    # Set random seed if provided
    if random_seed:
        np.random.seed(random_seed)

    # Shuffle the data
    shuffled_indices = np.random.permutation(len(sentences))
    shuffled_sentences = sentences[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    # Calculate the number of samples for training
    train_samples = int(len(sentences) * train_size)

    # Split the data into training and validation sets
    train_sentences = shuffled_sentences[:train_samples]
    train_labels = shuffled_labels[:train_samples]
    val_sentences = shuffled_sentences[train_samples:]
    val_labels = shuffled_labels[train_samples:]

    return train_sentences, train_labels, val_sentences, val_labels

train_size = 0.8  # 80% of the data for training

train_sentences, train_labels, val_sentences, val_labels = split_data_randomly(sentences, labels, train_size, random_seed=42)

print(len(train_sentences))
print(len(val_sentences))
print(len(train_labels))
print(len(val_labels))


# In[74]:


AUGMENTATION_SAMPLE_COUNT = 2
random.seed(42) 

def random_delete(sentence):
    """
    Pick a random index in a sentence, the word associated with that index is to be deleted.

    Arguments: 
        sentence: sentence string before random word removal
    Returns: 
        new_sentences: list of sentence strings after random word removal
    """
    words = sentence.split()
    new_sentences = []

    
    
    if len(words) > AUGMENTATION_SAMPLE_COUNT:
        
        # select one random word to remove for each new sample (3 new samples in total)
        random_indices =  np.random.choice(np.arange(0, len(words) + 1), size=AUGMENTATION_SAMPLE_COUNT , replace=False)

        # remove random word
        for random_index in random_indices:
            new_words = words.copy()
            try:
                new_words = words[:random_index] + words[random_index+1:]
                new_sentence = " ".join(new_words)

                # add to the list of new sentences based on the current sample
                new_sentences.append(new_sentence)
            except IndexError:
                pass
        
    return new_sentences



# In[75]:


def get_synonyms(word):
    """
    Given a word, use WordNet to get a list of synonyms
    """
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if word != l.name():
                synonyms.append(l.name())
    return synonyms



# In[76]:


AUGMENTATION_SAMPLE_COUNT = 2

def synonym_replace(sentence):
    """
    Pick a random index in a sentence. Replace the word associated with that index with a random synonym. 
    (Assumes that stopwords, punctuation have already been removed)

    Arguments: 
        sentence: sentence string before augmentation
    Returns: 
        new_sentences: list of sentence strings after augmentation
    """
    words = sentence.split()
    new_sentences = []

    
    
    if len(words) > AUGMENTATION_SAMPLE_COUNT :
        # select one random word to remove for each new sample 
        random_indices =  np.random.choice(np.arange(0, len(words) + 1), size=AUGMENTATION_SAMPLE_COUNT, replace=False)

        # remove random word
        for random_index in random_indices:
            new_words = words.copy()
            
            try:
                synonym = random.choice(get_synonyms(words[random_index]))
                new_words[random_index] = synonym
                new_sentence = " ".join(new_words)

                print(new_words)

                # add to the list of new sentences based on the current sample
                new_sentences.append(new_sentence)
            except IndexError:
                pass
        
    return new_sentences



# In[77]:


# Import translation model
from transformers import MarianMTModel, MarianTokenizer

# Helper function to download data for a language
def download(model_name):
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)
  return tokenizer, model

# download model for English -> Romance
tmp_lang_tokenizer, tmp_lang_model = download('Helsinki-NLP/opus-mt-en-ROMANCE')
# download model for Romance -> English
src_lang_tokenizer, src_lang_model = download('Helsinki-NLP/opus-mt-ROMANCE-en')


# In[78]:


def translate(texts, model, tokenizer, language):
  """Translate texts into a target language"""
  # Format the text as expected by the model
  formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
  original_texts = [formatter_fn(txt) for txt in texts]

  # Tokenize (text to tokens)
  tokens = tokenizer.prepare_seq2seq_batch(original_texts)

  # Translate
  translated = model.generate(**tokens)

  # Decode (tokens to text)
  translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

  return translated_texts

def back_translate(texts, language_src, language_dst):
  """Implements back translation"""
  # Translate from source to target language
  translated = translate(texts, tmp_lang_model, tmp_lang_tokenizer, language_dst)

  # Translate from target language back to source language
  back_translated = translate(translated, src_lang_model, src_lang_tokenizer, language_src)

  return back_translated


# In[79]:


# List for all the new sentences created through random deletion
# all_new_stcs = []

# # Loop through sentences in training data
# # If the sentence is a positive sample (minority class),
# # create new sentences by randomly selecting one word to remove for each new sentence
# for stc_idx, stc in enumerate(train_sentences):
#     if labels[stc_idx] == 1:
#         new_stcs_batch = synonym_replace(stc)

#         # Add the new sentences created from this individual sample to all new sentences list
#         all_new_stcs.extend(new_stcs_batch)

# # convert to numpy array and concatenate to previous sentence array
# array_all_new_stcs = np.array(all_new_stcs)
# train_sentences = np.concatenate((train_sentences, array_all_new_stcs), axis=0)

# # add n positive labels (n being the number of new sentences) to the labels array
# array_all_new_labels = np.array([1] * len(all_new_stcs))
# train_labels = np.concatenate((train_labels, array_all_new_labels), axis=0)

# print(len(train_sentences))
# print(len(val_sentences))
# print(len(train_labels))
# print(len(val_labels))

############ end of data augmentation ######################


# In[80]:


# This just shows how it has been tokenized and id-mapped. We can delete.
print(' Original: ', train_sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(train_sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_sentences[0]))) 


# In[81]:


# Training
# This actually tokenizes all the sentences and updates the max length so the model knows where to pad the sentences.
max_len = 0

# For every sentence...
for sent in train_sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    train_input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(train_input_ids))

print('Max sentence length: ', max_len)


# In[82]:


# Validation
# This actually tokenizes all the sentences and updates the max length so the model knows where to pad the sentences.
# max_len = 0

# For every sentence...
for sent in val_sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    val_input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(val_input_ids))

print('Max sentence length: ', max_len)


# In[83]:


train_input_ids = []
train_attention_masks = []

# For every stc...
for stc in train_sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    train_encoded_dict = tokenizer.encode_plus(
                        stc,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation=True,
                        #max_length = max_len,           # Pad & truncate all sentences.
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

    # Add the encoded sentence to the list.
    train_input_ids.append(train_encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    train_attention_masks.append(train_encoded_dict['attention_mask'])

# Convert the lists into tensors.
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', train_sentences[0])
print('Token IDs:', train_input_ids[0])


# In[84]:


val_input_ids = []
val_attention_masks = []

# For every stc...
for stc in val_sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    val_encoded_dict = tokenizer.encode_plus(
                        stc,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation=True,
                        #max_length = max_len,           # Pad & truncate all sentences.
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

    # Add the encoded sentence to the list.
    val_input_ids.append(val_encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    val_attention_masks.append(val_encoded_dict['attention_mask'])

# Convert the lists into tensors.
val_input_ids = torch.cat(val_input_ids, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
val_labels = torch.tensor(val_labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', val_sentences[0])
print('Token IDs:', val_input_ids[0])


# In[85]:


# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

# Combine the validation inputs into a TensorDataset.
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)


# Create a 90-10 train-validation split.

# # Calculate the number of samples to include in each set.
# train_size = int(0.8 * len(dataset))
# #val_size = int(0.2 * len(dataset))
# val_size = len(dataset)  - train_size

# # Divide the dataset by randomly selecting samples.
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# print('{:>5,} training samples'.format(train_size))
# print('{:>5,} validation samples'.format(val_size))

print(len(train_dataset), ' training samples')
print(len(val_dataset), ' validation samples')

print(train_dataset[:10])


# In[86]:


# # Combine the training inputs into a TensorDataset.
# dataset = TensorDataset(input_ids, attention_masks, labels)

# # Create a 90-10 train-validation split.

# # Calculate the number of samples to include in each set.
# train_size = int(0.8 * len(dataset))
# #val_size = int(0.2 * len(dataset))
# val_size = len(dataset)  - train_size

# # Divide the dataset by randomly selecting samples.
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# print('{:>5,} training samples'.format(train_size))
# print('{:>5,} validation samples'.format(val_size))


# In[87]:


# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
batch_size = 16

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# In[88]:


# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# enable gradient checkpointing to avoid out of memory with GPU training
model.gradient_checkpointing_enable()

# if device == "cuda:0":
# # Tell pytorch to run this model on the GPU.
#     model = model.cuda()
model = model.to(device)


# In[89]:


optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


# In[90]:


# Number of training epochs
epochs = 2

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# In[91]:


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[92]:


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[93]:


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the device using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        optimizer.zero_grad()
        output = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = output.loss
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables
    total_eval_accuracy = 0
    best_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            output= model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
        loss = output.loss
        total_eval_loss += loss.item()
        # Move logits and labels to CPU if we are using GPU
        logits = output.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    if avg_val_accuracy > best_eval_accuracy:
        torch.save(model, 'bert_model_EngWithStopwords')
        best_eval_accuracy = avg_val_accuracy
    #print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    #print("  Validation took: {:}".format(validation_time))
    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


# In[56]:


model = torch.load('bert_model_EngWithStopwords')


# In[58]:


df_test = pd.read_csv('data\\split_data\\eval_dataset.csv')

# for i in df_test['text']:
#     try: clean_text(i) 
#     except AttributeError:
#         print(i)

df_test['text'] = df_test['text'].apply(lambda x:clean_text(x))
test_sentences = df_test['text'].values


# In[59]:


test_input_ids = []
test_attention_masks = []
for stc in test_sentences:
    encoded_dict = tokenizer.encode_plus(
                        stc,
                        truncation=True,
                        padding = 'max_length',
                        add_special_tokens = True,
                        # max_length = max_len,
                        # pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    test_input_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])
test_input_ids = torch.cat(test_input_ids, dim=0) 
test_attention_masks = torch.cat(test_attention_masks, dim=0)


# In[61]:


test_dataset = TensorDataset(test_input_ids, test_attention_masks)
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# In[62]:


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


# In[63]:


df_output = pd.DataFrame()
#df_output['id'] = df_test['id'] # Do we need to add ids?
df_output['target'] =predictions
df_output.to_csv('D4_BERT_FINAL_EVAL.out',index=False, header=False)


# In[46]:


# EVALUATION

import sys
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd

results_filename = 'D4_BERT_NO_DATAAUG_withSW.out'
test_set_csv = 'data\\split_data\\dev_dataset.csv'

# read in the CSV and get the gold labels
gold_df = pd.read_csv(test_set_csv)
task1_gold = gold_df['target'].tolist()

# read in the results file and get the system output labels
task1_res = []
with open(results_filename) as f:
    for line in f:
        task1_res.append(int(line.strip()))

with open('D4_BERT_NO_DATAAUG_withSW_Results.txt', 'w') as outf:

    # task 1 scores
    t1p = precision_score(task1_gold, task1_res)
    t1r = recall_score(task1_gold, task1_res)
    t1f = f1_score(task1_gold, task1_res)
    # task1
    outf.write('task1_precision:'+str(t1p)+'\n')
    outf.write('task1_recall:'+str(t1r)+'\n')
    outf.write('task1_f1:'+str(t1f)+'\n')    


# In[ ]:




