import pandas as pd
import numpy as np

#import original dataset
data = pd.read_csv("Codes&datasets/nba_logreg.csv", header = "infer")

#replacing all missing values, if any, to NaN
data = data.replace('?',np.NaN)
data = data.replace('',np.NaN)

#showing the dimentions before everything
print(f'Number of instances before = {data.shape[0]}')
print(f'Number of attributes before = {data.shape[1]}')

#printing before
sum2 = 0
for col in data.columns:
    sum2 += data[col].isna().sum()
print(f"Number of missing values before: {sum2}")

#inserting missing values randomly
for i in range(16):
    rr = np.random.randint(0, data.shape[0])
    rc = np.random.randint(0, data.shape[1])
    data.iat[rr, rc] = np.NaN
#printing after
sum2 = 0
for col in data.columns:
    sum2 += data[col].isna().sum()
print(f"Number of missing values after: {sum2}")


#showing number of dups before
dups = data.duplicated()
print(f'Number of duplicate rows before = {dups.sum()}')

#inserting dups randomly
for i in range(8):
    rr = np.random.randint(0, data.shape[0])
    data = data._append(data.iloc[rr])

#showing dups after
dups = data.duplicated()
print(f'Number of duplicate rows after = {dups.sum()}')

#showing the dimentions after everything
print(f'Number of instances after = {data.shape[0]}')
print(f'Number of attributes after = {data.shape[0]}')


#export dataset after inserting missing values and duplicates
data.to_csv('Codes&datasets/nba_edited.csv', index=False)