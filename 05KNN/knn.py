import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =5 , metric ='minkowski')
classifier.fit(x_train,y_train)
print(classifier.score(x_test,y_test))
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
