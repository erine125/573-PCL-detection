#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home2/qywang/miniconda3/envs/pcl-env

pip install -r ../requirements.txt

python BERT_PCL_detection.py 
python Translate_and_Evaluate.py
python BERT_PCL_detection_Chinese.py 
python RoformerModel.py