# -*- coding: utf-8 -*-
"""
Hyperparameter Optimization using GridSearchCV and RandomizedSearchCV

@author: Aditi Bongale
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix


# Paths within directory
combined_dir = 'data/03_CombinedProcessedData_csv'
combined_csv_name = 'Combined_I1_4K.csv'
results_dir = "results"
results_csv_name = 'Combined_I1_4K_HPO.csv'

# Read the CSV file into a DataFrame
input_data = pd.read_csv(os.path.join(combined_dir, combined_csv_name))
print('Tabular data read')

# Seperate 3 inputs
df = input_data.drop(['Target','Group','Median_i_dc','Median_v_dut'], axis=1)

df1 = df.iloc[:, 250:]
df_i_dc = df1.iloc[:, :-250]
df_v_dut = df.iloc[:, 500:]

# Prepare data for model cross validation
X = pd.concat([df_i_dc, df_v_dut], axis=1)
y = input_data['Target']
groups = input_data['Group'].values 


# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 10, 30],
    'max_features': ['sqrt', 'log2']
}

fixed_seed = 42

# Outer loop: LOGO cross-validation for evaluation

logo = LeaveOneGroupOut()
outer_results = []
best_params_per_fold = []

for fold_i, (train_idx, test_idx) in enumerate (logo.split(X, y, groups)):
    # Split the data for this outer fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    outer_random_state = fixed_seed + fold_i

    # Inner loop: Grid search with LOGO cross-validation
    inner_logo = LeaveOneGroupOut()
    grid_search = GridSearchCV(
        estimator=ExtraTreesClassifier(random_state = outer_random_state),
        param_grid=param_grid,
        scoring = 'recall', # scoring metric
        cv=inner_logo.split(X_train, y_train, groups[train_idx]),
        n_jobs=-1,
        error_score="raise"
    )
    
    # # Perform RandomizedSearchCV
    # grid_search = RandomizedSearchCV(
    #     estimator=ExtraTreesClassifier(random_state=outer_random_state),
    #     param_distributions=param_grid,
    #     scoring='recall',  # scoring metric
    #     cv=inner_logo.split(X_train, y_train, groups[train_idx]),
    #     n_iter=50,  # Number of parameter settings sampled
    #     n_jobs=-1,
    #     random_state=fixed_seed,
    #     error_score="raise",
    # )    
    
    # Perform grid search, but check if any validation set has only one class
    valid_folds = []
    for inner_train_idx, inner_val_idx in inner_logo.split(X_train, y_train, groups=groups[train_idx]):
        # Check if the validation set contains only one class
        if len(np.unique(y_train.iloc[inner_val_idx])) == 1:
            print(f"Warning: Inner validation set contains only one class. Skipping fold.")
            continue  # Skip the fold if validation set has only one class
        valid_folds.append((inner_train_idx, inner_val_idx))  # Store valid folds

    if valid_folds:
        grid_search.fit(X_train, y_train)  # Perform the hyperparameter tuning
        # Best model from inner CV (this model has the best found hyperparameters)
        best_model = grid_search.best_estimator_
        # Store the best parameters for later analysis
        best_params_per_fold.append(grid_search.best_params_)
        # Train the best model on the outer training data and test on the outer test data
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        # Handle single-class case in confusion matrix for the outer fold
        cm = confusion_matrix(y_test, y_pred)
        if len(np.unique(y_test)) == 1:
            # Handle case where test set has only one class (either all positives or all negatives)
            if y_test.iloc[0] == 0:  # Only negatives in y_test
                tn = cm[0, 0]  # True Negatives
                fp = cm[0, 1] if cm.shape[1] > 1 else 0  # False Positives
                fn = 0
                tp = 0
            else:  # Only positives in y_test
                tp = cm[0, 0]  # True Positives
                fn = cm[0, 1] if cm.shape[1] > 1 else 0  # False Negatives
                tn = 0
                fp = 0
        else:
            # Normal case with mixed classes
            tn, fp, fn, tp = cm.ravel()

        # Compute evaluation metrics
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        # Store results for this outer fold
        outer_results.append({
            'confusion_matrix': (tn, fp, fn, tp),
            'fnr': fnr,
            'fpr': fpr,
            'recall': recall,
            'accuracy': accuracy
        })                

# Aggregate final results
average_results = {
    metric: np.mean([result[metric] for result in outer_results])
    for metric in ['fnr', 'fpr', 'recall', 'accuracy']
}
print("Average Metrics:")
print(average_results)

best_params_data = []
for i, best_params in enumerate(best_params_per_fold):
    fold_params = {"Fold": i + 1}
    fold_params.update(best_params)  # Add all hyperparameters for this fold
    best_params_data.append(fold_params)
best_params_df = pd.DataFrame(best_params_data)
best_params_df.to_csv("best_HPO_I1.csv", index=False)

detailed_results = []
for i, result in enumerate(outer_results):
    fold_result = {
        "Fold": i + 1,
        "Confusion Matrix": result['confusion_matrix'],  # Convert numpy array to list
        "FNR": result['fnr'],
        "FPR": result['fpr'],
        "Recall": result['recall'],
        "Accuracy": result['accuracy']
    }
    detailed_results.append(fold_result)

# Convert to DataFrame and save to CSV
results_df = pd.DataFrame(detailed_results)
results_df.to_csv("detailed_results_I1.csv", index=False)