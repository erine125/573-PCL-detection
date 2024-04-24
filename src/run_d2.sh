#!/bin/sh

pip install -r requirements.txt

/dropbox/23-24/WIN571/envs/571/bin/python split_reformat_data.py data/dontpatronizeme_v1.5/DPM_trainingset/dontpatronizeme_pcl.tsv 0.9

/dropbox/23-24/WIN571/envs/571/bin/python GloVe_PCL_detection.py /dropbox/23-24/WIN571/hw9/glove.6B.50d.txt data/split_data/train_dataset.csv data/split_data/dev_dataset.csv ../outputs/D2/Glove_ConvNN_withUndersampling_output.txt 

/dropbox/23-24/WIN571/envs/571/bin/python BERT_PCL_detection.py data/split_data/train_dataset.csv data/split_data/dev_dataset.csv ../outputs/D2/BERT_2epochs_output.txt 

/dropbox/23-24/WIN571/envs/571/bin/python evaluation.py ../outputs/D2/Glove_ConvNN_withUndersampling_output.txt data/split_data/dev_dataset.csv ../results/D2/Glove_ConvNN_withUndersampling_output.txt

/dropbox/23-24/WIN571/envs/571/bin/python evaluation.py ../outputs/D2/BERT_2epochs_output.txt data/split_data/dev_dataset.csv ../results/D2/BERT_2epochs_output.txt
