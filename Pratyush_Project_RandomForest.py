
# Importing Librarires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('churn_prediction.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Filling Up Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:,0:17])
X[:,0:17] = imputer.transform(X[:,0:17])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer.fit(X[:,17:19])
X[:,17:19] = imputer.transform(X[:,17:19])


# Encoding Categorical Data (If Any)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 17] = labelencoder_X_1.fit_transform(X[:, 17])
labelencoder_X_2 = LabelEncoder()
X[:, 18] = labelencoder_X_1.fit_transform(X[:, 18])
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [18])], remainder='passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:].astype(float)


# Splitting The Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training The Classification Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicitng Results
y_pred = classifier.predict(X_test)

#Creating The Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
