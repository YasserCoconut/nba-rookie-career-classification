import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import pydotplus 
from IPython.display import Image
import matplotlib.pyplot as plt
#=====================================================================================================================================================
#In this python file, we will apply the decision tree cassifier

data = pd.read_csv("Codes&datasets/nba_afterPreProc.csv", header = "infer")


# Put our target class in a variable
Y = data["TARGET_5Yrs"]
# Put the dataset in a variable called X and drop 'Name' and the target column
X = data.drop(["Name", "TARGET_5Yrs"],axis=1)

depth = 12
# Make the decition tree with depth 10 (defined above)
TC = tree.DecisionTreeClassifier(criterion='entropy', max_depth = depth)
TC = TC.fit(X, Y)

# Make our decision tree into a picture
dot_data = tree.export_graphviz(TC, feature_names=X.columns, filled=True, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 

# Store our DT in a png file
graph.write_png("decision_tree.png")

# Display the DT
plt.figure(figsize=(15, 10))
plt.imshow(plt.imread("decision_tree.png"))
plt.axis("off")
plt.show()


#=====================================================================================================================================================
# In this section, we will use the Decision Tree created on a test dataset to find the accuracy:

# Load our test dataset into a testData variable
testData = data

# Store our target column in variable 'testY'
testY = testData["TARGET_5Yrs"]
# Drop 'Name' and target column from the dataset and store it in variable 'testX'
testX = testData.drop(["Name","TARGET_5Yrs"], axis=1)

# Use the predict() dunction and predict the target column from our test dataset and store it in 'predY'
predY = TC.predict(testX)
# Concatinating the datasets (combining them) and printing it
predictions = pd.concat([testData['Name'],pd.Series(predY,name='Predicted Class')], axis=1)
print(predictions)

# Printing the accuracy of our DT
print(f"Accuracy on test data with decision tree of depth {depth} is {accuracy_score(testY, predY)*100}%")

#=====================================================================================================================================================
