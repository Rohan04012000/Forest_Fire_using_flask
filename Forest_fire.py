import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

dataframe = pd.read_csv("Forest_fire_dataset.csv")
dataframe = np.array(dataframe)

X_data = dataframe[1:, 1:-1]
y_data = dataframe[1:, -1]
y_data = y_data.astype('int')
X_data = X_data.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 0)
Log_model = LogisticRegression()

Log_model.fit(X_train, y_train)

pickle.dump(Log_model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
