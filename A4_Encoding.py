import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Read our training and test data 
# training_data = pd.read_csv(str(sys.argv[1]), sep='\t')
# test_data = pd.read_csv(str(sys.argv[2]), sep='\t')

training_data = pd.read_csv('A4_TrainData.tsv', sep='\t', header=None)
test_data = pd.read_csv('A4_TestData.tsv', sep='\t', header=None)

# Clean up data and prepare for encoding
training_data.rename(columns={0:"Nominal Data", 1:'Class'}, inplace=True)

X = training_data['Nominal Data']
y = training_data['Class']

X_training = X.str.split('', expand=True)
X_training.replace('', float('NaN'), inplace=True)
X_training = X_training.dropna(how='all', axis=1)
X_training = X_training.rename(columns={X_training.columns[i-1]:i-1 for i in X_training.columns})

X_testing = test_data[0].str.split('', expand=True)
X_testing.replace('', float('NaN'), inplace=True)
X_testing = X_testing.dropna(how='all', axis=1)
X_testing = X_testing.rename(columns={X_testing.columns[i-1]:i-1 for i in X_testing.columns})

'''
This function takes in a classifier, an encoder and training data (X, y) and
creates a pipeline that encodes all the input data and creates a model.
The model is then tested using Stratified 10-Fold CV and the resulting PR curves
are plotted, along with the average precision score. 
'''
def plot_PRC_model(classifier, encoder, X, y):

  y_true = []
  y_proba = []
  p_scores = []

  ax = plt.subplot()

  for i, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=10).split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    column_transformer = make_column_transformer((encoder, X_train.columns))
    pipe = make_pipeline(column_transformer, classifier)
    probas = pipe.fit(X_train, y_train).predict_proba(X_test)

    precision, recall, _ = precision_recall_curve(y_test, probas[:, 1])
    p_score = average_precision_score(y_test, probas[:, 1])

    ax.plot(recall, precision, lw=1, alpha=0.3, label=f'PR fold {i} (AUC = {p_score})')

    y_true.append(y_test)
    y_proba.append(probas[:, 1])
    p_scores.append(p_score)


  y_true = np.concatenate(y_true)
  y_proba = np.concatenate(y_proba)

  precision, recall, _= precision_recall_curve(y_true, y_proba)

  ax.plot(recall, precision, color='r', lw=2, alpha=0.8, label=f'Mean PR (AUC = {average_precision_score(y_true, y_proba)} \u00B1 {np.std(p_scores)})')
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  ax.title.set_text(f'Precsion-Recall Curce with 10-Fold CV using {encoder} encoding')
  plt.show()
  
  pass
  
# Create a Random Forest Classifier object
classifier = RandomForestClassifier(n_estimators=2000, max_features='log2', class_weight='balanced')

# Create a One Hot Encoder object
one_hot_encoder = OneHotEncoder(sparse=False, dtype=int)

# Create an Ordinal Encoder object 
ordinal_encoder = OrdinalEncoder()

# Plot the PR curve for the Ordinal Encoder
Ord_model = plot_PRC_model(classifier, ordinal_encoder, X_training, y)

# Plot the PR curve for the One Hot Encoder 
OHE_model = plot_PRC_model(classifier, one_hot_encoder, X_training, y)


# Create and fit the final model
column_transformer = make_column_transformer((one_hot_encoder, X_training.columns))
pipe = make_pipeline(column_transformer, classifier)
pipe.fit(X_training, y)

# Predict the probablility of belonging to class 1
probas = pipe.predict_proba(X_testing)
probas_class_1 = probas[:, 1]