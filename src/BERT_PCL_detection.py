import pandas as pd
import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
from nltk.corpus import stopwords
import re
import sys
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import perturbation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# arguments:
# train_data_file: path to training data csv 
# test_data_file: path to test (or dev) data csv 
# pred_output_file: filename to print output to 
[_, train_data_file, test_data_file, pred_output_file] = sys.argv

# hyperparameters - can set up in config file later
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
UNDERSAMPLE = False
RANDOM_DELETION = True


SEED = 42

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Read data as PD dataframe
    df = pd.read_csv(train_data_file)

    # undersampling to balance the two classes
    if UNDERSAMPLE: 
        df = undersampling(df)

    
    # Apply preprocessing
    df['text'] = df['text'].apply(lambda x: clean_text(x))

    sentences = df.text.values
    labels = df.target.values

    # Apply data augmentation through random deletion for the minority class
    if RANDOM_DELETION:
        for stc_idx, stc in enumerate(sentences):
            if labels[stc_idx] == 1:
                new_stcs = random_delete(stc)
                sentences.extend(new_stcs)
                # what is the type of the elements in labels? int or string? int
                new_labels = [1] * len(new_stcs)
                labels.extend(new_labels)
            

    # tokenize and get max length
    max_len = 0
    for sent in sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    input_ids = []
    attention_masks = []

    for stc in sentences:
        encoded_dict = tokenizer.encode_plus(
                            stc,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            truncation=True,
                            #max_length = max_len,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Do a 80-20 train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset)  - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for train and validation datasets
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = BATCH_SIZE # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = BATCH_SIZE # Evaluate with this batch size.
            )
    
    # For our model, we will use BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Start with the pretrained BERT model with uncased vocab
        num_labels = 2, # The number of output labels--2 for binary classification.
        output_attentions = False, 
        output_hidden_states = False
    )

    # enable gradient checkpointing to avoid out of memory with GPU training
    model.gradient_checkpointing_enable()

    # move to CUDA device
    model = model.to(device)

    # set optimizer
    optimizer = AdamW(model.parameters(),
                  lr = LEARNING_RATE, 
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * NUM_EPOCHS

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    for epoch_i in range(0, NUM_EPOCHS):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, NUM_EPOCHS))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Move to CUDA device
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        # Validation
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
            torch.save(model, 'bert_model')
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
    

    # load model (if necessary)
    model = torch.load('bert_model')


    df_test = pd.read_csv(test_data_file)

    # DEBUG
    # for i in df_test['text']:
    #     try: clean_text(i) 
    #     except AttributeError:
    #         print(i)

    df_test['text'] = df_test['text'].apply(lambda x:clean_text(x))
    test_sentences = df_test['text'].values

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

    test_dataset = TensorDataset(test_input_ids, test_attention_masks)
    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    
    # get predictions
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


    # write predictions to outfile 
    with open(pred_output_file, 'w') as outfile:
        for prediction in predictions:
            outfile.write(str(prediction) + '\n')



# -----------------------------------------------------
# HELPER FUNCTIONS

def undersampling(data, target_column='target'):
    # Split the data into positive and negative samples
    positive_samples = data[data[target_column] == 1]
    negative_samples = data[data[target_column] == 0]
    
    num_positive = positive_samples.shape[0]
    
    # Sample from the negative samples to get a random subset equal to the number of positive samples
    negative_subset = negative_samples.sample(n=num_positive, random_state=SEED) 
    balanced_data = pd.concat([positive_samples, negative_subset])
    
    # Shuffle the dataset to mix positive and negative samples
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_data


# ----------------------------------------------------
# The following functions are adapted from the following tutorial by Neeraj Mohan:
# https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification 


def clean_text(text):
    """
    Removes punctuation, stopwords, URLs, emojis
    """
    sw = stopwords.words('english')

    text = text.lower()

    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    text = re.sub(r"http\S+", "",text) #Removing URLs
    #text = re.sub(r"http", "",text)

    html=re.compile(r'<.*?>')

    text = html.sub(r'',text) #Removing html tags

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') #Removing punctuations

    text = [word.lower() for word in text.split() if word.lower() not in sw]

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

def get_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

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

    new_words = words.copy()
    
    # change this to select 9 random ints all at once
    random_indices =  np.random.choice(np.arange(0, len(words) + 1), size=9, replace=False)
    for random_index in random_indices:
        try:
            del new_words[random_index]
            new_sentence = new_words.join(" ")
            new_sentences.append(new_sentence)
        except IndexError:
            pass
        
    return new_sentences


main()