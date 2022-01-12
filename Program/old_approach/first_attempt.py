import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def read_data(source_filename, target_filename):
    if os.path.exists(target_filename):
        data = pd.read_csv(target_filename)
    else:
        # read data
        data = pd.read_csv(source_filename).dropna()
        # extract duration time
        data['DURATION'] = (pd.to_datetime(data['CMPLNT_TO_TIMESTAMP']) -
                            pd.to_datetime(data['CMPLNT_FR_TIMESTAMP'])
                            ).map(lambda delta: delta.total_seconds())
        # drop not necessary columns
        data.drop('CMPLNT_NUM', axis=1, inplace=True)
        data.drop('RPT_TIMESTAMP', axis=1, inplace=True)
        data.drop('CMPLNT_TO_TIMESTAMP', axis=1, inplace=True)
        data.drop('CMPLNT_FR_TIMESTAMP', axis=1, inplace=True)
        # encode categorical columns
        encoder = LabelEncoder()
        data['SUSP_AGE_GROUP'] = encoder.fit_transform(data['SUSP_AGE_GROUP'])
        data['SUSP_RACE'] = encoder.fit_transform(data['SUSP_RACE'])
        data['SUSP_SEX'] = encoder.fit_transform(data['SUSP_SEX'])
        data['VIC_AGE_GROUP'] = encoder.fit_transform(data['VIC_AGE_GROUP'])
        data['VIC_RACE'] = encoder.fit_transform(data['VIC_RACE'])
        data['VIC_SEX'] = encoder.fit_transform(data['VIC_SEX'])
        data['PREM_TYP_DESC'] = encoder.fit_transform(data['PREM_TYP_DESC'])
        data['LOC_OF_OCCUR_DESC'] = encoder.fit_transform(
            data['LOC_OF_OCCUR_DESC'])
        data['BORO_NM'] = encoder.fit_transform(data['BORO_NM'])
        data['LAW_CAT_CD'] = encoder.fit_transform(data['LAW_CAT_CD'])
        data['CRM_ATPT_CPTD_CD'] = encoder.fit_transform(
            data['CRM_ATPT_CPTD_CD'])
        # save preprocessed data
        data.to_csv(target_filename)
    return np.array(data.drop('DURATION', axis=1)), np.array(data['DURATION'])


if __name__ == '__main__':
    print("Preprocessing data...")
    X, y = read_data("data/NYPD_Data_Preprocessed-204442.csv",
                     "data/NYPD_Data_Preprocessed-204442_time_regression.csv")
    print("\t DONE")

    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("\t DONE")

    for max_depth in [4, 5, 6, 7, 10, 20]:
        print(f"Training (max_depth={max_depth})...")
        regressor = RandomForestRegressor(max_depth=max_depth,
                                          n_jobs=-1,
                                          verbose=1)
        regressor.fit(X_train, y_train)
        print("\t DONE")

        print("Testing...")
        y_pred_train = regressor.predict(X_train)
        y_pred_test = regressor.predict(X_test)
        print(f"\t TRAIN mean absolute error {mean_absolute_error(y_train, y_pred_train)}")
        print(f"\t TEST mean absolute error {mean_absolute_error(y_test, y_pred_test)}")
        print("\t DONE")
