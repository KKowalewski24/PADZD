import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("./NYPD_Complaint_Data_Historic_preprocessed.csv", index_col=0)


def one_hot_encode(data: pd.DataFrame, column_name: str, drop=None):
    encoder = OneHotEncoder(sparse=True, dtype=np.uint8, drop=drop)
    encoded = pd.DataFrame.sparse.from_spmatrix(
        encoder.fit_transform(data[[column_name]]),
        columns=encoder.get_feature_names_out(),
    )
    data = pd.concat([data, encoded], axis=1)
    data.drop(column_name, axis=1, inplace=True)
    return data


# encoding
data = one_hot_encode(data, "KY_CD")
data = one_hot_encode(data, "LAW_CAT_CD")
data = one_hot_encode(data, "CRM_ATPT_CPTD_CD", drop=["ATTEMPTED"])
data = one_hot_encode(data, "LOC_OF_OCCUR_DESC", drop=[np.nan])
data = one_hot_encode(data, "PREM_TYP_DESC", drop=[np.nan])
data = one_hot_encode(data, "VIC_SEX", drop=[np.nan])
data = one_hot_encode(data, "SUSP_SEX", drop=[np.nan])
data = one_hot_encode(data, "VIC_RACE", drop=[np.nan])
data = one_hot_encode(data, "SUSP_RACE", drop=[np.nan])
data["VIC_AGE_GROUP"] = (
    data["VIC_AGE_GROUP"]
    .map({"<18": 0, "18-24": 1, "25-44": 2, "46-64": 3, "65+": 4})
    .astype(np.float16)
)
data["SUSP_AGE_GROUP"] = (
    data["VIC_AGE_GROUP"]
    .map({"<18": 0, "18-24": 1, "25-44": 2, "46-64": 3, "65+": 4})
    .astype(np.float16)
)

# train/test split
print("Splitting to train and test datasets...")
data = data.sample(frac=1, random_state=47).reset_index(drop=True)
y = data["CMPLNT_SECOND_OF_DAY"].astype("uint32")
X = data.drop("CMPLNT_SECOND_OF_DAY", axis=1)
n_test_samples = int(0.3 * len(data))

# save to DMatrix
print("Loading data to DMatrix...")
test_data = xgb.DMatrix(X[:n_test_samples], label=y[:n_test_samples])
train_data = xgb.DMatrix(X[n_test_samples:], label=y[n_test_samples:])

# save to disk
print("Saving binary data to disk...")
test_data.save_binary("test_data.buffer")
train_data.save_binary("train_data.buffer")
