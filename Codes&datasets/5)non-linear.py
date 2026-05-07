import pandas as pd
from sklearn.model_selection import train_test_split
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

# Variables for storing results
C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
SVMtrainAcc = []
SVMtestAcc = []

for param in C:
    #SVM with non-linear
    clf = SVC(C=param,kernel='rbf',gamma='auto')
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    SVMtrainAcc.append(accuracy_score(Y_train, Y_predTrain))
    SVMtestAcc.append(accuracy_score(Y_test, Y_predTest))

#=====================================================================================================================================================
# Plotting the results
plt.plot(C, SVMtrainAcc, 'ro-', C, SVMtestAcc, 'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.title('SVM Accuracy for Different C Values')
plt.xlabel('C (Regularization Parameter)')
plt.xscale('log')
plt.ylabel('Accuracy')

# Display the plot
plt.show()