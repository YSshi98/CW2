import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


training_data = pd.read_csv("TrainingDataMulti.csv")
testing_data = pd.read_csv("TestingDataMulti.csv")