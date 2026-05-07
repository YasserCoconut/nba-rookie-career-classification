from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
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

#=====================================================================================================================================================
# Variables for storing results
numBaseClassifiers = 500
maxdepth = 12
trainAcc = []
testAcc = []
#=====================================================================================================================================================
#Random Forest classifier method
clf = ensemble.RandomForestClassifier(n_estimators=numBaseClassifiers)
clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
Y_predTest = clf.predict(X_test)
trainAcc.append(accuracy_score(Y_train, Y_predTrain))
testAcc.append(accuracy_score(Y_test, Y_predTest))
#=====================================================================================================================================================
# Bagging classifier method
clf = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=maxdepth),n_estimators=numBaseClassifiers)
clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
Y_predTest = clf.predict(X_test)
trainAcc.append(accuracy_score(Y_train, Y_predTrain))
testAcc.append(accuracy_score(Y_test, Y_predTest))
#=====================================================================================================================================================
# AsaBoost classifier method
clf = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=maxdepth),n_estimators=numBaseClassifiers)
clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
Y_predTest = clf.predict(X_test)
trainAcc.append(accuracy_score(Y_train, Y_predTrain))
testAcc.append(accuracy_score(Y_test, Y_predTest))
#=====================================================================================================================================================
# Plotting the results
methods = ['Random Forest', 'Bagging', 'AdaBoost']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.bar(methods, trainAcc, color=['blue', 'orange', 'green'])
ax1.set_title('Training Accuracy')
ax1.set_ylabel('Accuracy')
ax1.grid(axis='y', linestyle='--', linewidth=0.5)

ax2.bar(methods, testAcc, color=['blue', 'orange', 'green'])
ax2.set_title('Test Accuracy')
ax2.set_ylabel('Accuracy')
ax2.grid(axis='y', linestyle='--', linewidth=0.5)

plt.show()