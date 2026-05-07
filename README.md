# NBA Rookie Career Longevity Classification

This project explores and compares several machine learning classification techniques using Python and Scikit-learn. The goal is to predict whether an NBA rookie will remain in the league for at least 5 years based on rookie-season performance statistics.

The project was completed for a Data Mining course and demonstrates a full classification workflow, including dataset preparation, preprocessing, exploratory analysis, model training, evaluation, and result visualization.

## Project Overview

The dataset contains performance attributes for NBA rookie players. The target variable is `TARGET_5Yrs`, where:

- `0` = the player played fewer than 5 years
- `1` = the player played 5 years or more

The project compares the performance of multiple classification models:

- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Linear Support Vector Machine (SVM)
- Nonlinear SVM using RBF kernel
- Random Forest
- Bagging Classifier
- AdaBoost Classifier

## Main Features

- Loads and prepares an NBA rookie dataset for binary classification
- Introduces missing values and duplicate rows for preprocessing practice
- Handles missing values, duplicates, and outliers
- Balances the target classes to reduce class bias
- Uses cross-tabulation to explore relationships between features and the target class
- Trains and evaluates multiple classification models
- Visualizes model results using plots and accuracy comparisons
- Generates and saves a visual Decision Tree model

## Dataset

The dataset used in this project is `nba_logreg.csv`. It contains NBA rookie statistics such as games played, field goals made, and other performance indicators.

The project includes three dataset versions:

```text
Codes&datasets/nba_logreg.csv        # Original dataset
Codes&datasets/nba_edited.csv        # Dataset after inserting missing values and duplicates
Codes&datasets/nba_afterPreProc.csv  # Dataset after preprocessing
```

A backup copy of the datasets is also included in the `BackupDatasets` folder.

## Project Structure

```text
Project/
│
├── Codes&datasets/
│   ├── 0)InsertingMissingValues&dups.py
│   ├── 1)preProcessing.py
│   ├── 2)DT.py
│   ├── 3)K-nn.py
│   ├── 4)linear.py
│   ├── 5)non-linear.py
│   ├── 6)ensemble.py
│   ├── nba_logreg.csv
│   ├── nba_edited.csv
│   ├── nba_afterPreProc.csv
│   └── README.txt
│
├── BackupDatasets/
│   ├── nba_logreg.csv
│   ├── nba_edited.csv
│   └── nba_afterPreProc.csv
│
├── CrossTab(FGM vs Target).png
├── CrossTab(FGM&GP vs Target).png
├── CrossTab(GP vs Target).png
├── decision_tree.png
├── LogisticRegression&SVM(linear).png
├── SVM Accuracy for different C values(non-linear).png
├── TrainingAccuracy&TestAccuracy for emsemble.png
└── 2023_FA_BCS_304_Project_Rubric.docx
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn pydotplus ipython
```

You may also need Graphviz installed on your system to generate the Decision Tree image.

### 3. Run the scripts in order

The scripts are numbered from `0` to `6`. Run them in this order:

```bash
python "Codes&datasets/0)InsertingMissingValues&dups.py"
python "Codes&datasets/1)preProcessing.py"
python "Codes&datasets/2)DT.py"
python "Codes&datasets/3)K-nn.py"
python "Codes&datasets/4)linear.py"
python "Codes&datasets/5)non-linear.py"
python "Codes&datasets/6)ensemble.py"
```

Important note: running scripts `0` and `1` will modify/regenerate the edited and preprocessed datasets. If you want to keep the current dataset and generated charts unchanged, do not run scripts `0` and `1`.

To restore the original project datasets, copy the CSV files from `BackupDatasets` into `Codes&datasets` and replace the existing files.

## Model Summary

### Decision Tree Classifier

A Decision Tree model was trained using entropy as the splitting criterion. Different tree depths were tested, and a depth of 12 was selected as a balance between accuracy and computational efficiency. The final tree was visualized and saved as `decision_tree.png`.

### K-Nearest Neighbors

The KNN model was trained with `n_neighbors=5`. It was evaluated using accuracy, precision, recall, F1-score, and support.

### Linear Models

Logistic Regression and Linear SVM were tested using different values of the regularization parameter `C`. The training and testing accuracy values were plotted to compare performance.

### Nonlinear SVM

A nonlinear SVM using the RBF kernel was trained with different values of `C`. The results showed how changes in regularization can affect training and testing accuracy.

### Ensemble Methods

Random Forest, Bagging, and AdaBoost classifiers were implemented and compared using training and testing accuracy. These models showed strong training performance, while the testing results helped identify possible overfitting.

## Results and Visualizations

The project includes several generated visualizations:

- Cross-tabulation between Field Goals Made and target class
- Cross-tabulation between Games Played and target class
- Cross-tabulation between Field Goals Made, Games Played, and target class
- Decision Tree visualization
- Logistic Regression and Linear SVM accuracy comparison
- Nonlinear SVM accuracy across different `C` values
- Ensemble method training and testing accuracy comparison

## Key Takeaways

- Preprocessing is essential before training classification models, especially when missing values, duplicate rows, outliers, and class imbalance are present.
- Decision Trees are easy to interpret and performed strongly on this dataset, especially when tree depth was tuned.
- KNN and SVM models required careful parameter selection and generally produced lower testing accuracy than the Decision Tree in this project.
- Ensemble models achieved high training accuracy but showed signs of overfitting when testing accuracy was much lower.
- Comparing multiple classifiers helps identify the strengths and weaknesses of each approach for a specific dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pydotplus
- IPython Display

## Possible Improvements

Future improvements could include:

- Using cross-validation for more reliable model evaluation
- Applying feature scaling consistently across all models
- Tuning hyperparameters using GridSearchCV or RandomizedSearchCV
- Adding confusion matrices and ROC curves
- Improving reproducibility by setting random seeds in all scripts
- Organizing scripts into a cleaner pipeline or notebook format
- Adding a `requirements.txt` file for easier setup

## Author

Created as part of a Data Mining course project on classification techniques using Scikit-learn.
