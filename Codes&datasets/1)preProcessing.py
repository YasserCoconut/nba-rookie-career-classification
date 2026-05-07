import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the edited dataset that contains missing values and dups
data = pd.read_csv("Codes&datasets/nba_edited.csv", header = "infer")

# Display the number of instances and attributes
print(f'Number of instances = {data.shape[0]}')
print(f'Number of attributes = {data.shape[1]}')


# Display the first few rows of the dataset
print(data.head())

#=====================================================================================================================================================
# This section takes care of missing values int he dataset:

# Check for missing values
print(f"Number of missing values in each column: ")
for col in data.columns:
    print(f'\t{col}: {data[col].isna().sum()}')

# Printing rows before discarding about missing values
print(f'Number of rows before discarding missing values = {data.shape[0]}')

data = data.dropna()

# Check for missing values after dropping the rows of missing values.
print('Number of missing values in each column:')
for col in data.columns:
    print(f'\t{col}: {data[col].isna().sum()}')

# Printing rows after taking care of missing values
print(f'Number of rows after discarding missing values = {data.shape[0]}')


#=====================================================================================================================================================
# This section takes care of duplicated data in the dataset:

#Checking the number of duplicated data
dups = data.duplicated()
print(f'Number of duplicate rows = {dups.sum()}')


# Printing rows before discarding about duplicates
print(f'\nNumber of rows before discarding about duplicates = {data.shape[0]}')

data = data.drop_duplicates()

# Printing rows after taking care of duplicates
print(f'Number of rows after discarding duplicates = {data.shape[0]}')

#=====================================================================================================================================================
# This section takes care of outliers:

# Printing rows before discarding outliers
print(f'\nNumber of rows before discarding outliers = {data.shape[0]}')

numeric_columns = data.select_dtypes(include=[np.number]).columns

# Identify the first quartile (Q1) and third quartile (Q3) for numeric columns
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)

# Calculate the IQR for each numeric column
IQR = Q3 - Q1

# Set a threshold for considering a data point as an outlier (you can adjust this)
threshold = 1.5

# Create a boolean mask indicating outliers for numeric columns
outliers_mask_numeric = (
    (data[numeric_columns] < (Q1 - threshold * IQR)) | 
    (data[numeric_columns] > (Q3 + threshold * IQR))
)

# Combine the outlier mask for numeric columns with a mask for non-numeric columns
outliers_mask = outliers_mask_numeric.any(axis=1)

# Remove rows containing outliers
data= data[~outliers_mask]

# Printing rows after discarding outliers
print(f'Number of rows after discarding outliers = {data.shape[0]}')

#=====================================================================================================================================================
# This section balances the target column so that it will not be biased; to prevent overfitting towards the majority class.

#Printing rows before balancing
print(f'\nNumber of rows before balancing = {data.shape[0]}')


# Putting the target majority and minority in variables.
T_majority = data[data['TARGET_5Yrs'] == 1]
T_minority = data[data['TARGET_5Yrs'] == 0]

# Swapping the majority and minority if minority is more than majority
if T_majority.shape[0] < T_minority.shape[0]:
    T_majority, T_minority = T_minority, T_majority

# Dropping random rows in T_majority till T_majority and T_minority have the same length
rows_to_drop = np.random.choice(T_majority.index, size=T_majority.shape[0] - T_minority.shape[0], replace=False)
T_majority = T_majority.drop(rows_to_drop)

# Combining T_majority and T_minority... Now we have a balanced dataset.
data = pd.concat([T_majority, T_minority])

# Printing rows after balancing
print(f'\nNumber of rows after balancing = {data.shape[0]}')

#=====================================================================================================================================================
# In this section, we do cross-tabulation to explore relationships between attributes and classes:


# Cross tabulation between Field Goals Made vs target class
crosstab_result = pd.crosstab(data["Field Goals Made"], data["TARGET_5Yrs"])
crosstab_result.plot(kind='bar', stacked=True)
plt.xlabel('Field Goals Made')
plt.ylabel('Count')
plt.title('Field Goals Made vs. TARGET_5Yrs')
plt.show()

# Cross tabulation between Games Played vs target class
crosstab_result = pd.crosstab(data["Games Played"], data["TARGET_5Yrs"])
crosstab_result.plot(kind='bar', stacked=True)
plt.xlabel('Games Played')
plt.ylabel('Count')
plt.title('Games Played vs. TARGET_5Yrs')
plt.show()

# Cross tabulation between (Field Goals Made & Games Played) vs target class
crosstab_result = pd.crosstab([data["Field Goals Made"], data["Games Played"]], data["TARGET_5Yrs"])
# You may choose a suitable visualization for two categorical variables
# For example, a heatmap:
crosstab_result.plot(kind='bar', stacked=True)
plt.xlabel('Field Goals Made, Games Played')
plt.ylabel('Count')
plt.title('Field Goals Made and Games Played vs. TARGET_5Yrs')
plt.show()

#=====================================================================================================================================================

# Save the csv file into a new one after pre processing.
data.to_csv('Codes&datasets/nba_afterPreProc.csv', index=False)
