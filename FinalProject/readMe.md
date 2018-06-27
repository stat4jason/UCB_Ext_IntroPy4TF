This is the folder for Zhicheng (Jason) Xue and Shiwen (GiGi) Wang's final project for X433.7 machine learning with TensorFlow.


Please note that Part V in the code is the section where you can find our TensorFlow models while the other codes in front are either data preparations or scikit learn model.


In order to run the program, you need to get all the packages listed in the code(Final_Project_X4337.py) and data(loan.csv). We suggeste to ues Jupiter notebook to run the code to avoid certain errors. If you want to compare against our results, please check the uploaded Jupyter Notebook file and all the pictures in the folder.

All the modules needed:
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly
plotly.offline.init_notebook_mode()
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn import preprocessing
from tensorflow.contrib.factorization import KMeans
import os

