"""
Decision Tree implementation on MNIST dataset

"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv('/kaggle/input/mnist_train.csv')
test_df  = pd.read_csv('/kaggle/input/mnist_test.csv')

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Train model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Evaluate
y_pred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
