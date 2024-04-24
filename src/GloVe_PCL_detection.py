import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import nltk
import sys
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch.utils.data import random_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# arguments:
# glove_embeddings_file: path to the file containing the glove embeddings
# train_data_file: path to training data csv 
# test_data_file: path to test (or dev) data csv 
# pred_output_file: filename to print output to 
[_, glove_embeddings_file, train_data_file, test_data_file, pred_output_file] = sys.argv

# hyperparameters - can set up in config file later
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
UNDERSAMPLE = True 
SEED = 42

def main():
    # load GloVe embeddings from text file 
    with open(glove_embeddings_file, encoding="utf8") as infile:
            lines = infile.readlines() 

    glove_embeddings = {} # dictionary that maps words to embeddings (NP arrays)

    for line in lines:
        line = line.strip().split() 
        word = line[0]
        nums = [float(x) for x in line[1:]]
        vector = torch.FloatTensor(nums)
        glove_embeddings[word] = vector 

    # Read data as PD dataframe
    df = pd.read_csv(train_data_file)

    # undersampling to balance the two classes
    if UNDERSAMPLE: 
        df = undersampling(df)

    # clean text to remove punctuation, stopwords, etc.
    df['text'] = df['text'].apply(lambda x: clean_text(x))

    # tokenize and get lists of text and targets
    train_sentences = df.text.values
    train_labels = df.target.values
    train_sentences = [word_tokenize(sentence) for sentence in train_sentences]

    dataset = get_vectors(glove_embeddings, train_sentences, train_labels)

    # split train and validation data 80-10
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset)  - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create the DataLoaders for our training and validation sets.
    train_loader = torch.utils.data.DataLoader(
                train_dataset,  # The training samples.
                sampler = torch.utils.data.RandomSampler(train_dataset), # Select batches randomly
                batch_size = BATCH_SIZE 
            )

    valid_loader = torch.utils.data.DataLoader(
                val_dataset, # The validation samples.
                sampler = torch.utils.data.SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = BATCH_SIZE 
            )
    

    # define our model 

    # Initial attempt using a linear model with ReLU 
    # model = nn.Sequential(nn.Linear(50, 40),
    #                     nn.ReLU(),
    #                     nn.Linear(40, 20),
    #                     nn.ReLU(),
    #                     nn.Linear(20, 2))


    # Second attempt using a Conv1D Layer with MaxPooling
    model = nn.Sequential(nn.Dropout(p=0.2),
                      nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1),
                      nn.MaxPool1d(2),
                      nn.Flatten(),
                      nn.Linear(5*25, 2)
                     )

    # Move model to CUDA device
    model = model.to(device)

    train_network(model, train_loader, valid_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    # Read, clean, and tokenize test data
    test_df = pd.read_csv(test_data_file)
    test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))
    test_sentences = test_df.text.values
    test_labels = test_df.target.values
    test_sentences = [word_tokenize(sentence) for sentence in test_sentences]

    test_dataset = get_vectors(glove_embeddings, test_sentences, test_labels)

    # DEBUG
    # print(test_sentences[0])
    # print(test_dataset[0])

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # test over devset
    predictions = []

    for lines, targets in test_loader:
        lines, targets = lines.to(device), targets.to(device) 
        output = model(lines)
        pred = output.max(1, keepdim=True)[1]

        predictions.extend(list(pred.flatten()))

    predictions = [pred.item() for pred in predictions]


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

def get_vectors(glove_vector, sentences, labels):
    """
    Given the GloVe vectors, gets the vectors for each sentence
    """
    dataset = []
    for i, line in enumerate(sentences):
        vector_sum = torch.zeros(50)
        if len(line) > 0:
            for w in line:
                # if i == 0:
                #     print(w)
                #     print(glove_vector[w])
                #     print(len(line))
                if w in glove_vector: # ignore out of vocabulary words for now
                    vector_sum += glove_vector[w] 
    
            aggregate_vector = vector_sum / len(line) # use mean as a sentence aggregate (sentence2vec)
            aggregate_vector = aggregate_vector.view(1, 50)
                
        label = torch.tensor(labels[i]).long()
        dataset.append((aggregate_vector, label))
            
    return dataset

# ------------------------------------------
# The following training and accuracy functions are adapted from the following tutorial by Lisa Zhang from UToronto CS:
# https://www.cs.toronto.edu/~lczhang/aps360_20191/lec/w06/sentiment.html 


def train_network(model, train_loader, valid_loader, num_epochs=5, learning_rate=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):
        for lines, targets in train_loader:
            lines, targets = lines.to(device), targets.to(device) 

            # DEBUG
            # print("lines")
            # print(lines.size())
            # print("targets")
            # print(targets.size())
            
            optimizer.zero_grad()
            pred = model(lines)

            # DEBUG
            # print("Output shape:", pred.shape)  
            # print("Target shape:", targets.shape)  
            
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))     

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train_loader))
        valid_acc.append(get_accuracy(model, valid_loader))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
            epoch+1, loss, train_acc[-1], valid_acc[-1]))

    # plotting
    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, valid_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

def get_accuracy(model, data_loader):
    correct, total = 0, 0
    for lines, targets in data_loader:
        lines, targets = lines.to(device), targets.to(device)
        
        output = model(lines)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(targets.view_as(pred)).sum().item()
        total += targets.shape[0]
    return correct / total

main()