# -*- coding: utf-8 -*-
"""
Model Training - LOGO Cross validation technique

@author: Aditi Bongale
"""

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import LeaveOneGroupOut

#=======================================================
#Function to validate the model : LOGO ; for Inverter 1
#=======================================================
def validate_model_LOGO_I1(X, y, groups, mixed):
    
    # Set a fixed seed for reproducibility
    fixed_seed = 42
    
    # To store the scores for each split
    accs, precs, recalls, f1_scores = [], [], [], []
    roc_aucs, prc_aucs, fprs, fnrs, tnrs = [], [], [], [], []
    
    # To store confusion matrix values for each fold
    tn_values, fp_values, fn_values, tp_values = [], [], [], []
    
    # To store the predictions for evaluation later
    groupleftout, y_preds, y_tests = [], [], []

    # To store the train and prediction time for each split
    train_ts, predict_ts = [], []
    
    # Initialize LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    
    # Perform LeaveOneGroupOut cross-validation
    for fold_i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Generate a new random state for each iteration
        random_state = fixed_seed + fold_i
        
        start_t = time.time()
        # Initialize and fit model on train
        et_classifier = ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1)
        et_classifier.fit(X_train, y_train)
        end_t = time.time()
        train_t = end_t - start_t
        
        # Make predictions
        start_p = time.time()
        y_pred = et_classifier.predict(X_test)
        end_p = time.time()
        predict_t = end_p - start_p
        
        # Probability scores for ROC/PRC
        y_pred_proba = et_classifier.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Check if both classes are present in y_test before calculating AUC and confusion matrix
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            prc_auc = average_precision_score(y_test, y_pred_proba)
            
            # Confusion matrix and derived metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            roc_auc = None  # or np.nan
            prc_auc = None  # or np.nan
            
            # Handle single-class case in confusion matrix
            cm = confusion_matrix(y_test, y_pred)
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
            
            # Derived metrics
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            
        # Append metrics to lists
        accs.append(accuracy)
        precs.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        prc_aucs.append(prc_auc)
        fprs.append(fpr)
        fnrs.append(fnr)
        tnrs.append(tnr)
        
        # Append confusion matrix values
        tn_values.append(tn)
        fp_values.append(fp)
        fn_values.append(fn)
        tp_values.append(tp)
            
        groupleftout.append(np.unique(groups[test_index]))
        y_preds.append(y_pred)
        y_tests.append(y_test)
            
        # Append time for each split/group
        train_ts.append(train_t)
        predict_ts.append(predict_t)
    
    average_recall = np.sum(recalls)/mixed
    average_accuracy = np.mean(accs)
    average_fnr = np.sum(fnrs)/mixed
    average_fpr = np.mean(fprs)
    print(f'Average Accuracy: {average_accuracy:.4f}')
    print(f'Average FNR: {average_fnr:.4f}')
    print(f'Average FPR: {average_fpr:.4f}')
    print(f'Average Recall: {average_recall:.4f}')
    print('\n Extra Trees Classification done')

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Group Left Out': groupleftout,
        'TP':tp_values,
        'FP':fp_values,
        'TN':tn_values,
        'FN':fn_values,
        'TNR/Specificity': tnrs,
        'FNR': fnrs,
        'FPR': fprs,
        'Accuracy': accs,
        'TPR/Recall': recalls,
        'Precision': precs,
        'F1 Score': f1_scores,
        'ROC_AUC': roc_aucs,
        'PRC_AUC': prc_aucs,
        'Train Time': train_ts,
        'Predict Time': predict_ts
    })       
    return results_df, y_preds, y_tests

#=======================================================
#Function to validate the model : LOGO ; for Inverter 2
#=======================================================
def validate_model_LOGO_I2(X, y, groups, mixed):
    
    # Set a fixed seed for reproducibility
    fixed_seed = 42
    
    # To store the scores for each split
    accs, precs, recalls, f1_scores = [], [], [], []
    roc_aucs, prc_aucs, fprs, fnrs, tnrs = [], [], [], [], []
    
    # To store confusion matrix values for each fold
    tn_values, fp_values, fn_values, tp_values = [], [], [], []
    
    # To store the predictions for evaluation later
    groupleftout, y_preds, y_tests = [], [], []

    # To store the train and prediction time for each split
    train_ts, predict_ts = [], []
    
    # Initialize LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    
    # Perform LeaveOneGroupOut cross-validation
    for fold_i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Generate a new random state for each iteration
        random_state = fixed_seed + fold_i
        
        start_t = time.time()
        # Initialize and fit model on train
        et_classifier = ExtraTreesClassifier(
            n_estimators=300,
            max_leaf_nodes=15000,
            random_state=random_state,
            n_jobs=-1)
        et_classifier.fit(X_train, y_train)
        end_t = time.time()
        train_t = end_t - start_t
        
        # Make predictions
        start_p = time.time()
        y_pred = et_classifier.predict(X_test)
        end_p = time.time()
        predict_t = end_p - start_p
        
        # Probability scores for ROC/PRC
        y_pred_proba = et_classifier.predict_proba(X_test)[:, 1]
        
        # Use for classification threshold of 0.28
        # y_pred = (y_pred_proba >= 0.28).astype(int) 
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Check if both classes are present in y_test before calculating AUC and confusion matrix
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            prc_auc = average_precision_score(y_test, y_pred_proba)
            
            # Confusion matrix and derived metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            roc_auc = None  # or np.nan
            prc_auc = None  # or np.nan
            
            # Handle single-class case in confusion matrix
            cm = confusion_matrix(y_test, y_pred)
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
            
            # Derived metrics
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            
        # Append metrics to lists
        accs.append(accuracy)
        precs.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        prc_aucs.append(prc_auc)
        fprs.append(fpr)
        fnrs.append(fnr)
        tnrs.append(tnr)
        
        # Append confusion matrix values
        tn_values.append(tn)
        fp_values.append(fp)
        fn_values.append(fn)
        tp_values.append(tp)
            
        groupleftout.append(np.unique(groups[test_index]))
        y_preds.append(y_pred)
        y_tests.append(y_test)
            
        # Append time for each split/group
        train_ts.append(train_t)
        predict_ts.append(predict_t)
    
    average_recall = np.sum(recalls)/mixed
    average_accuracy = np.mean(accs)
    average_fnr = np.sum(fnrs)/mixed
    average_fpr = np.mean(fprs)
    print(f'Average Accuracy: {average_accuracy:.4f}')
    print(f'Average FNR: {average_fnr:.4f}')
    print(f'Average FPR: {average_fpr:.4f}')
    print(f'Average Recall: {average_recall:.4f}')
    print('\n Extra Trees Classification done')

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Group Left Out': groupleftout,
        'TP':tp_values,
        'FP':fp_values,
        'TN':tn_values,
        'FN':fn_values,
        'TNR/Specificity': tnrs,
        'FNR': fnrs,
        'FPR': fprs,
        'Accuracy': accs,
        'TPR/Recall': recalls,
        'Precision': precs,
        'F1 Score': f1_scores,
        'ROC_AUC': roc_aucs,
        'PRC_AUC': prc_aucs,
        'Train Time': train_ts,
        'Predict Time': predict_ts
    })       
    return results_df, y_preds, y_tests
