from __future__ import absolute_import, print_function, division
from time import time
import pandas as pd
from six.moves import range
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import *
import matplotlib.pyplot as plt

class MICE():

    def __init__(
            self,
            model=LinearRegression(),
            init_fill_method="mean",
            min_value=None,
            max_value=None,
            verbose=True):
        self.min_value=min_value,
        self.max_value=max_value,
        self.fill_method=init_fill_method
        self.model = model
        self.verbose = verbose

    def perform_imputation_round(
            self,
            X_filled,
            missing_mask,
            observed_mask):
        n_rows, n_cols = X_filled.shape
        n_missing_for_each_column = missing_mask.sum(axis=0)
        ordered_column_indices = np.arange(n_cols)
        for col_idx in range(len(X.T)):
            missing_row_mask_for_this_col = missing_mask[:, col_idx]
            n_missing_for_this_col = n_missing_for_each_column[col_idx]
            if n_missing_for_this_col > 0: 
                observed_row_mask_for_this_col = observed_mask[:, col_idx]
                column_values = X_filled[:, col_idx]
                #print(column_values)
                column_values_observed = column_values[observed_row_mask_for_this_col]
                other_column_indices = np.concatenate([
                        ordered_column_indices[:col_idx],
                        ordered_column_indices[col_idx + 1:]
                    ])
                X_other_cols = X_filled[:, other_column_indices]
                #print(X_other_cols)
                #print(X_other_cols)
                X_other_cols_observed = X_other_cols[observed_row_mask_for_this_col]
                #print(X_other_cols_observed)
                X_other_cols_missing = X_other_cols[missing_row_mask_for_this_col]
                #print(X_other_cols_missing)
            
                lr = self.model
                lr.fit(X_other_cols_observed,column_values_observed)
                X_other_cols_missing = X_other_cols[missing_row_mask_for_this_col]
                y1 = lr.predict(X_other_cols_missing)
                #print(y2)
                X_filled[missing_row_mask_for_this_col, col_idx] = y1
        return X_filled 
        
    def initialize(self, X, missing_mask, observed_mask):
        X_filled = X.copy()
        for col_idx in range(len(X.T)):
            missing_mask_col = missing_mask[:, col_idx]
            n_missing = missing_mask_col.sum()
            if n_missing > 0:
                observed_row_mask_for_col = observed_mask[:, col_idx]
                column = X_filled[:, col_idx]
                observed_column = column[observed_row_mask_for_col]
                if self.fill_method == "mean":
                    fill_values = np.mean(observed_column)
                else:
                    raise ValueError("Invalid fill method %s" % self.fill_method)
                X_filled[missing_mask_col, col_idx] = fill_values
        return X_filled

    def multiple_imputations(self, X):
        start_t = time()
        X = np.asarray(X)
        missing_mask = np.isnan(X)
        missing_mask = np.asarray(missing_mask)
        observed_mask = ~missing_mask
        X_filled = self.initialize(
            X,
            missing_mask=missing_mask,
            observed_mask=observed_mask)
        #print(X_filled)
        results_list = []
        total_rounds = 10
        for m in range(total_rounds):
            print("[MICE] Starting imputation round %d/%d, elapsed time %0.3f" % (
                        m + 1,
                        total_rounds,
                        time() - start_t))
            X_filled = self.perform_imputation_round(
                X_filled=X_filled,
                missing_mask=missing_mask,
                observed_mask=observed_mask)
            results_list.append(X_filled[missing_mask])
        return np.array(results_list), missing_mask

    def complete(self, X):
        print("[MICE] Completing matrix with shape %s" % (X.shape,))
        X_completed = np.array(X.copy())
        imputed_arrays, missing_mask = self.multiple_imputations(X)
        #print(imputed_arrays)
        average_imputated_values = imputed_arrays.mean(axis=0)
        X_completed[missing_mask] = average_imputated_values
        return X_completed

imputer = MICE()
X=pd.read_csv(r'C:\Users\admin_\Desktop\dataset.csv')
X=X.replace(0,np.nan)
#print(X)
Y=imputer.complete(X)
a=np.float32(Y)
#plt.plot(a)
#plt.show()
print(a)
#np.savetxt(r"C:\Users\admin_\Desktop\accc3.csv",a)