# 573-project

This repository contains source code for our system for [SemEval 2022 Task 4: Patronizing and Condescending Language Detection](https://aclanthology.org/2022.semeval-1.38/). This system was built as a project for LING 573: NLP Systems & Applications at the University of Washington. 

First, install necessary requirements: `pip install -r requirements.txt`

We have already split the data into the train and dev sets, available under `src/data/split_data`. If you wish to run the data splitting and reformatting code, run: `python /src/split_reformat_data.py /data/dontpatronizeme_v1.5/DPM_trainingset/dontpatronizeme_pcl.tsv <devset_percentage>` (we used 0.1 for devtest percentage). Otherwise, you can use the existing files `src/data/split_data/train_dataset.csv` and `src/data/split_data/dev_dataset.csv` as the train and test files when running the model scripts.

Our initial models are in `src/GloVe_PCL_detection.py` and `src/BERT_PCL_detection.py`. `src/GloVe_PCL_detection.py` and `src/BERT_PCL_detection.ipynb` contain the same code with some additional comments in a Jupyter Notebook file. 

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
Outputs for both models are available in `outputs/D2/`. Both models were trained locally on GPU, but are set to run on CPU if GPU is not available. 

Finally, to evaluate system output and generate the results files, run: `python /src/evaluation.py <results_filename> <devset_csv_filename> <outfile_name>`
This will output the precision, recall, and F1 score over the devset, in the same format as the official scorer for SemEval-2022.

Please see `src/run_d2.sh` for more specifics on the calling order and the arguments we use to run our system end-to-end.

Our final D4 system can be run end-to-end using the .ipynb or .py files. The files are:
* `src/BERT_PCL_detection.[ipynb | py]` for the primary task 
* `src/Translate_and_Evaluate.[ipynb | py]` for the adaptation task (translating the Chinese data, then evaluating on the pretrained model saved from BERT_PCL_detection
* `src/BERT_PCL_Detection_Chinese.[ipynb | py]` for the adaptation task (direct training on Chinese data)
* `src/RoformerModel.[ipynb | py]` for the adaptation task using RoFormer model

Each of the files except for `Translate_and_Evaluate` handle training, testing, and evaluation end-to-end, and output both an output (`.out`) and results (`.txt`) file. 

