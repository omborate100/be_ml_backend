import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_csv('StartupData.csv')
x = df.drop(['got_offer(1/0)'], axis=1)
y = df['got_offer(1/0)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

log = LogisticRegression()
log.fit(x_train, y_train)
x1 = log.predict(x_test)
print(accuracy_score(y_test, x1))
print(x_train)

pickle.dump(log, open('model1.pkl', 'wb'))
model = pickle.load(open('model1.pkl', 'rb'))
