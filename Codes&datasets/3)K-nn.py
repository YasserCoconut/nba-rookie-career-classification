from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load our dataset
data = pd.read_csv("Codes&datasets/nba_afterPreProc.csv", header="infer")

# X is the feature matrix, Y is the target variable
Y = data["TARGET_5Yrs"]
# Drop 'Name' and target column from the dataset and store it in variable X
X = data.drop(["Name", "TARGET_5Yrs"], axis=1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a k-NN classifier (you can adjust the value of 'n_neighbors')
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier on the training data
knn_classifier.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)*100
print(f'Accuracy: {accuracy:.2f}%')

# Print a classification report for more detailed metrics
print(classification_report(Y_test, y_pred))
