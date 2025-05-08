# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset (use any dataset, here we use the Iris dataset again)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, header=None, names=columns)

# Preprocessing the data
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Target variable (class labels)

# Convert categorical target to numerical
y = pd.get_dummies(y).values  # One-hot encoding

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better training performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Predict on the test set
y_pred = dt.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

print("\nClassification Report:")
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

# Accuracy
accuracy = dt.score(X_test, y_test)
print(f'\nAccuracy: {accuracy*100:.2f}%')

# Visualize the Decision Tree (Optional)
plt.figure(figsize=(12, 8))
tree.plot_tree(dt, feature_names=columns[:-1], class_names=dataset['class'].unique(), filled=True)
plt.show()
