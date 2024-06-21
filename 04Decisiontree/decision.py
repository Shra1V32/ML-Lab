import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


import warnings 
warnings.filterwarnings('ignore')



df = pd.read_csv("adult_dataset.csv")

df.info()
df.head()

  
df=df[df['workclass']!= '?']
df=df[df['occupation']!='?']
df=df[df['native.country']!='?']

 
from sklearn import preprocessing
df_categorical = df.select_dtypes(include=['object'])
df_categorical.head()

 
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()

df=df.drop(df_categorical.columns,axis=1)
df = pd.concat([df,df_categorical], axis =1)


df.info()


from sklearn.model_selection import train_test_split
x=df.iloc[: , :-1].values
y=df.iloc[: ,-1].values 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)



from sklearn.tree import DecisionTreeClassifier
df_default = DecisionTreeClassifier(max_depth=5)
df_default.fit(x_train,y_train)

from sklearn.metrics  import classification_report, confusion_matrix, accuracy_score
y_pred_default = df_default.predict(x_test)
print(classification_report(y_test,y_pred_default))
print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test,y_pred_default))
df_default.score(x_test,y_test)

df_gini = DecisionTreeClassifier(criterion='gini',max_depth=10, min_samples_leaf=50,min_samples_split=50)
df_gini.fit(x_train,y_train)
print(df_gini.score(x_test,y_test))
