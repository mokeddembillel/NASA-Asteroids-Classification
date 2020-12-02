# Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Reading the Data
dataset = pd.read_csv('./Data/nasa.csv', sep=',', na_values=["n/a", "na", "--"])

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Cheking Missing Values
dataset.isnull().any()



