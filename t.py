import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import numba as nb
import sklearn
import cv2
import os

from time_checker import frequency_based, frequency_based_scipy, edge_based, laplacian_variance, time_it

def classify_edge(lst:list):
    treshold = 5
    return [0 if i >= treshold else 1 for i in lst]

if __name__ == '__main__':
    df = pd.read_csv('processed_results.csv', index_col=False)
    
    y_true = np.array(df['label'])
    y_scores = classify_edge(df['edge_based'])
    print(classification_report(y_true, y_scores))
