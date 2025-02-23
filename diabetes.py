import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold

dataset=pd.read_csv('C:/Users/adars/OneDrive/Desktop/Yukti-project/Diabetic/diabetes.csv')
dataset.head()

dataset['Outcome'].value_counts()

dataset.groupby('Outcome').mean()

dataset.shape

dataset.describe()

dataset.isnull().sum()

x=dataset.drop(columns='Outcome', axis=1)
y=dataset['Outcome']
print(x)
print(y)

scaler=StandardScaler()
scaler.fit(x)

standardized_data=scaler.transform(x)
print(standardized_data)

A=standardized_data
B=dataset['Outcome']

print(A)
print(B)



x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=2)

print(x.shape,x_train.shape,x_test.shape)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train , y_train)

result=regressor.predict([[6,148,72,35,0,33.6,0.627,50]])
if result==1:
   print("The person is diabetic")
else:
   print("The person is not diabetic")
   print(result)

#Model Evaluation

