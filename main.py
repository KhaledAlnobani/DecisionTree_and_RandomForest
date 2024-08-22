import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree, accuracy
from RandomForest import RandomForest

df = pd.read_csv('winequality-red.csv')
X = df.drop('quality', axis=1)
y = df['quality']

# Convert to binary classification: Good (6-8) vs Bad (3-5)
y = (y >= 7).astype(int)

# Label encoding for the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = np.array(X)
y = np.array(y)

# Split data into training ,testing and cross validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_cv, y_test, y_cv = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

max_depth = 20
clf = DecisionTree(max_depth=max_depth)
clf.fit(X_train, y_train)

test_predictions = clf.predict(X_test)
test_accuracy = accuracy(test_predictions, y_test)
print("Testing Accuracy for decision tree:", test_accuracy)

cv_predictions = clf.predict(X_cv)
cv_accuracy = accuracy(cv_predictions, y_cv)
print("CV Accuracy for decision tree:", cv_accuracy)

clf2 = RandomForest(max_depth=max_depth, B=200)
clf2.fit(X_train, y_train)

test_predictions = clf2.predict(X_test)
test_accuracy = accuracy(test_predictions, y_test)
print("Testing Accuracy for random forest without  randomizing features:", test_accuracy)

cv_predictions = clf2.predict(X_cv)
cv_accuracy = accuracy(cv_predictions, y_cv)
print("CV Accuracy for random forest without randomizing features:", cv_accuracy)

clf3 = RandomForest(max_depth=max_depth, randomizing_features=True, B=200)
clf3.fit(X_train, y_train)

test_predictions = clf3.predict(X_test)
test_accuracy = accuracy(test_predictions, y_test)
print("Testing Accuracy for random forest with  randomizing features:", test_accuracy)

cv_predictions = clf3.predict(X_cv)
cv_accuracy = accuracy(cv_predictions, y_cv)
print("CV Accuracy for random forest with randomizing features:", cv_accuracy)
