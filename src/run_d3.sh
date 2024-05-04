#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home2/qywang/miniconda3/envs/pcl-env

pip install -r ../requirements.txt

# python GloVe_PCL_detection.py /dropbox/23-24/WIN571/hw9/glove.6B.50d.txt data/split_data/train_dataset.csv data/split_data/dev_dataset.csv ../outputs/D3/Glove_ConvNN_withUndersampling_output.txt 

python BERT_PCL_detection.py data/split_data/train_dataset.csv data/split_data/dev_dataset.csv ../outputs/D3/BERT_2epochs_output.txt 

# python evaluation.py ../outputs/D3/Glove_ConvNN_withUndersampling_output.txt data/split_data/dev_dataset.csv ../results/D3/Glove_ConvNN_withUndersampling_output.txt

python evaluation.py ../outputs/D3/BERT_2epochs_output.txt data/split_data/dev_dataset.csv ../results/D3/BERT_2epochs_output.txt
