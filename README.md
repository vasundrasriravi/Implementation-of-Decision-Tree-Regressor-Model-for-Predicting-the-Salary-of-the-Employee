# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
Developed by: VASUNDRA SRI R
RegisterNumber: 212222230168 
```
```
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Salary.csv')
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform (data["Position"])
data.head()
x=data[["Position", "Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=2)
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score (y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```
## Output:
![Screenshot 2024-04-03 092248](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393983/792faadb-7dee-4515-a14f-ad80017974b1)

### MSE Value:
![Screenshot 2024-04-03 092323](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393983/c3ebf2f4-8be2-4839-b181-bffe2c74b632)

### R2 Value:
![Screenshot 2024-04-03 092333](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393983/caf36175-d411-4013-b6a1-c6e0e80b43ff)

### Predicted Value:
![Screenshot 2024-04-03 092357](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393983/24c4589b-33a6-44b2-b2eb-5aea4362976b)

### Result Tree:
![Screenshot 2024-04-03 092501](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393983/0daa26ca-dca9-4dd9-b322-168dcd857037)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
