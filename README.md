# 573-project
To run our system for D2:

First, install necessary requirements: `pip install -r requirements.txt`

We have already split the data into the train and dev sets, available under `src/data/split_data`. If you wish to run the data splitting and reformatting code, run: `python /src/split_reformat_data.py /data/dontpatronizeme_v1.5/DPM_trainingset/dontpatronizeme_pcl.tsv <devset_percentage>` (we used 0.1 for devtest percentage)

Our two models are in `GloVe_PCL_detection.py` and `BERT_PCL_detection.py`. 

For the GloVe model, the arguments are:
```
glove_embeddings_file: path to the file containing the glove embeddings
train_data_file: path to training data csv 
test_data_file: path to test (or dev) data csv 
pred_output_file: filename to print output to 
```

For the BERT model, the arguments are:
```
train_data_file: path to training data csv 
test_data_file: path to test (or dev) data csv 
pred_output_file: filename to print output to 
```

These will generate prediction output files, which are text/csv files that contain each prediction (0 or 1) over the test set on each line.

Finally, to evaluate system output and generate the results files, run: `python /src/evaluation.py <results_filename> <devset_csv_filename> <outfile_name>`
This will print the precision, recall, and F1 score over the devset. 

