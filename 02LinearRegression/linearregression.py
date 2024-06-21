import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("Salary_Data.csv")
x = np.array(np.array(dataset.iloc[:,-1].values, ndmin=2))
y = np.array(dataset.iloc[:,1].values)
x.shape, y.shape
x = x.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
def mse(actual, predicted):
    return np.mean((np.square(actual - predicted)))
mean_squared_error = mse(y_test, y_pred)
print("Mean Sqaured Error = {}".format(mean_squared_error))
print("Root Mean Sqaured Error(RMSE) ={}".format(mean_squared_error**0.5))
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title('Salary vs Experience(Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
