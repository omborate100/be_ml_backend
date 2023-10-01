import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df= pd.read_csv('StartupData.csv')
# df.nunique(axis=0)

x = df.drop(['got_offer(1/0)'],axis=1)
y= df['got_offer(1/0)']

from sklearn.model_selection import train_test_split
x_train, x_test , y_train ,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2 )
classifier.fit(x_train, y_train)

##picke file
pickle.dump(classifier, open('model.pkl', "wb"))