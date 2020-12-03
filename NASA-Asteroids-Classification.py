# Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Reading the Data
dataset = pd.read_csv('./Data/nasa.csv', sep=',', na_values=["n/a", "na", "--"])

# Cheking Missing Values
dataset.isnull().any()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Encoding Object columns in the right format
X['Close Approach Date'] = pd.to_datetime(X['Close Approach Date'])
X['Orbiting Body'] = LabelEncoder().fit_transform(X['Orbiting Body'])
X['Orbit Determination Date'] = pd.to_datetime(X['Orbit Determination Date'])
X['Equinox'] = LabelEncoder().fit_transform(X['Orbiting Body'])
y = LabelEncoder().fit_transform(y)

X.info()

























