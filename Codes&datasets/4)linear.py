import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Loading our dataset and spliting it into training and testing sets
data = pd.read_csv("Codes&datasets/nba_afterPreProc.csv", header="infer")


# X is the feature matrix, Y is the target variable
Y = data["TARGET_5Yrs"]
# Drop 'Name' and target column from the dataset and store it in variable X
X = data.drop(["Name", "TARGET_5Yrs"], axis=1)
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the features using StandadScaler() from sklearn library
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Variables for storing results
C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
LRtrainAcc = []
LRtestAcc = []
SVMtrainAcc = []
SVMtestAcc = []

# Logistic Regression and SVM with linear
for param in C:
    # Logistic Regression
    clf = linear_model.LogisticRegression(C=param, max_iter=1000)
    clf.fit(X_train_normalized, Y_train)
    Y_predTrain = clf.predict(X_train_normalized)
    Y_predTest = clf.predict(X_test_normalized)
    LRtrainAcc.append(accuracy_score(Y_train, Y_predTrain))
    LRtestAcc.append(accuracy_score(Y_test, Y_predTest))
    

    # SVM with linear
    clf = SVC(C=param, kernel='linear')
    clf.fit(X_train_normalized, Y_train)
    Y_predTrain = clf.predict(X_train_normalized)
    Y_predTest = clf.predict(X_test_normalized)
    SVMtrainAcc.append(accuracy_score(Y_train, Y_predTrain))
    SVMtestAcc.append(accuracy_score(Y_test, Y_predTest))

#=====================================================================================================================================================
# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(C, LRtrainAcc, 'ro-', C, LRtestAcc, 'bv--')
ax1.legend(['Training Accuracy', 'Test Accuracy'])
ax1.set_xlabel('C')
ax1.set_xscale('log')
ax1.set_ylabel('Accuracy')
ax1.set_title('Logistic Regression')

ax2.plot(C, SVMtrainAcc, 'ro-', C, SVMtestAcc, 'bv--')
ax2.legend(['Training Accuracy', 'Test Accuracy'])
ax2.set_xlabel('C')
ax2.set_xscale('log')
ax2.set_ylabel('Accuracy')
ax2.set_title('SVM with Linear')

plt.show()
