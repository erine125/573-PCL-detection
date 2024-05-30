#!/usr/bin/env python
# coding: utf-8

# In[32]:


from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizerFast, RoFormerModel
import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
from transformers import RoFormerForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast = False, tokenize_chinese_chars =False)
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

# model = RoFormerModel.from_pretrained("junnyu/roformer_chinese_base")


# In[33]:


import torch
import pandas as pd

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split


# In[34]:


# to run on GPU with CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[35]:


df = pd.read_csv("data\\CCPC_Dataset.csv")
df.head(n=10)


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SEED = 42

train_data, test_data = train_test_split(df, train_size = 0.1, random_state = SEED)


# In[37]:


train_sentences = test_data.text.values
train_labels = df.target.values


# In[38]:


# split train into train and validation

import numpy as np

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

train_sentences, train_labels, val_sentences, val_labels = split_data_randomly(train_sentences, train_labels, train_size, random_seed=42)


# In[39]:


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


# In[40]:


# This just shows how it has been tokenized and id-mapped. We can delete.
print(' Original: ', train_sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(train_sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_sentences[0]))) 


# In[41]:


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


# In[42]:


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


# In[43]:


# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

# Combine the validation inputs into a TensorDataset.
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

print(len(train_dataset), ' training samples')
print(len(val_dataset), ' validation samples')

print(train_dataset[:10])


# In[44]:


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


# In[45]:


# model = RoFormerForSequenceClassification.from_pretrained(
#     "junnyu/roformer_chinese_base", 
#     num_labels = 2, # The number of output labels--2 for binary classification.
#                     # You can increase this for multi-class tasks.
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
# )

model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", # Use the 12-layer BERT model, with an uncased vocab.
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


# In[46]:


optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


# In[47]:


# Number of training epochs
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# In[48]:


import random
import datetime
import time


# In[49]:


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[50]:


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[51]:


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
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
        torch.save(model, 'bert_model_Chinese')
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


# In[ ]:


# TESTING

model = torch.load('bert_model_Chinese') # load the chinese model


# In[ ]:


# test_sentences = test_data['text'].values
# test_labels = test_data['target'].values

test_sentences = val_sentences 
test_labels = val_labels


# In[ ]:


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


# In[ ]:


batch_size = 16

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
results_filename = 'bert_chinese.out'

df_output.to_csv(results_filename,index=False, header=False)


# In[ ]:


# EVALUATION

import sys
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd

# read in the CSV and get the gold labels
task1_gold = test_labels

# read in the results file and get the system output labels
task1_res = []
with open(results_filename) as f:
    for line in f:
        task1_res.append(int(line.strip()))

with open('bert_chinese_dev_results.txt', 'w') as outf:

    # task 1 scores
    t1p = precision_score(task1_gold, task1_res)
    t1r = recall_score(task1_gold, task1_res)
    t1f = f1_score(task1_gold, task1_res)
    # task1
    outf.write('task1_precision:'+str(t1p)+'\n')
    outf.write('task1_recall:'+str(t1r)+'\n')
    outf.write('task1_f1:'+str(t1f)+'\n')    


# In[ ]:




