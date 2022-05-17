# Importing Packages and Modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

prospect_data = pd.read_csv(r'F:\shivahari\trimester 4\MACHINE LEARNING ALGORITHMS\CIA- 1\TermLife.csv')

#Descriptive Analysis
prospect_data.columns
prospect_data.describe()

#To take the values of all the rows and columns (except dependent variables)
matrix_feature_termlife = prospect_data.iloc[:, :-5].values

#To take the values of all the rows and columns (dependent variables)
dep_vector_y = prospect_data.iloc[:, -5].values

null_treater = SimpleImputer(missing_values=np.nan, strategy='mean')
null_treater.fit(matrix_feature_termlife[:, :-5])

# To replace all Null values
matrix_feature_termlife[:, :-5] = null_treater.transform(matrix_feature_termlife[:, :-5])


#In case of transforming any field to categorical data, bottom code can be used in future to transform it into binary vectors

#Encoding the categorical data (Transformed numerical data into category and creating binary vectors
#to differentiate each category and avoid specific patterns in columns

  # ct = ColumnTransformer(transformers=[('encoding', OneHotEncoder(), [column to be transformed])], remainder= 'passthrough')
  # matrix_feature_termlife = np.array(ct.fit_transform(matrix_feature_termlife))
  # #Need to force the output of my matrix features into a numpy array, in order to be used by future machine learning models
  # matrix_feature_termlife = np.array(ct.fit_transform(matrix_feature_termlife))

#Splitting my dataset into training set and test set

matrix_feature_train, matrix_feature_test, dep_vector_ytrain, dep_vector_ytest = train_test_split(matrix_feature_termlife, dep_vector_y, test_size=0.2, random_state=1)

#Feature Scaling (To have all values in same range) - Appling Standadisation to my Matrix of Features

sc = StandardScaler()
matrix_feature_train[:, 3:] = sc.fit_transform(matrix_feature_train[:, 3:])
matrix_feature_test[:, 3:] = sc.transform(matrix_feature_test[:, 3:])

#Training my simple linear regression model on training set
#To find out whether age of the respondents are related to the face amount

X = prospect_data.iloc[:, 1:-16].values
y = prospect_data.iloc[:, 13:-4].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
mod1 = regressor.score(X_train, y_train) * 100
print("Regression fit = ", mod1)
coeff1 = regressor.coef_
print("Regression coefficient =", coeff1)
int1 = regressor.intercept_
print("Regression intercept =", int1)
# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('AGE v/s FACE VALUE (Training set)')
plt.xlabel('AGE')
plt.ylabel('FACE VALUE')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('AGE v/s FACE VALUE (Test set)')
plt.xlabel('AGE')
plt.ylabel('FACE VALUE')
plt.show()

#INSIGHTS
#From our plot It's evident that AGE is showing a linear relationship with FACE amount eventhough there are many outliers
#Therefore we can infer that as AGE increases people are more prone to get insurance with a greater face amount.

#2
#To find out whether no. of household members of the respondents
# are related to the face amount?

rrr = prospect_data.iloc[:, 9:-8, ].values
yyy = prospect_data.iloc[:, 13:-4].values

reg2 = LinearRegression()
reg2.fit(X_train, y_train)
mod2 = reg2.score(X_train, y_train) * 100
print("Regression fit = ", mod2)
coeff2 = reg2.coef_
print("Regression coefficient =", coeff2)
int2 = regressor.intercept_
print("Regression intercept =", int2)
# Predicting the Test set results
y_pred2 = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'orange')
plt.title('NUMHH v/s FACE VALUE (Training set)')
plt.xlabel('NUMHH')
plt.ylabel('FACE VALUE')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('NUMHH v/s FACE VALUE (Test set)')
plt.xlabel('NUMHH')
plt.ylabel('FACE VALUE')
plt.show()
#INSIGHTS
#From the regression model, it's evident that number of household members is positively related to face
# value of insured amount and also the relationship is not strong

