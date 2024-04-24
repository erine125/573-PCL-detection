import sys
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd

"""
This code is a modified version of the official scorer for SemEval 2022, which is available online at:
https://github.com/Perez-AlmendrosC/dontpatronizeme/blob/master/semeval-2022/evaluation.py 
"""

# pass as arguments the files containing the results and the original CSV containing the gold output, and the name of the outfile to write the results to
[_, results_filename, test_set_csv, outfile_name] = sys.argv

# read in the CSV and get the gold labels
gold_df = pd.read_csv(test_set_csv)
task1_gold = gold_df['target'].tolist()

# read in the results file and get the system output labels
task1_res = []
with open(results_filename) as f:
    for line in f:
        task1_res.append(int(line.strip()))

with open(outfile_name, 'w') as outf:

    # task 1 scores
    t1p = precision_score(task1_gold, task1_res)
    t1r = recall_score(task1_gold, task1_res)
    t1f = f1_score(task1_gold, task1_res)
    # task1
    outf.write('task1_precision:'+str(t1p)+'\n')
    outf.write('task1_recall:'+str(t1r)+'\n')
    outf.write('task1_f1:'+str(t1f)+'\n')    

# Print the F1 score
# print('F1 score for: ' + results_filename + " " +str(t1f)+'\n')