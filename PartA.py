import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


training_data = pd.read_csv("TrainingDataBinary.csv")
testing_data = pd.read_csv("TestingDataBinary.csv")