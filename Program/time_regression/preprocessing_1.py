import numpy as np
import pandas as pd

data = pd.read_csv("./NYPD_Complaint_Data_Historic.csv")


# INDEX
# - not used at all
data.drop("CMPLNT_NUM", axis=1, inplace=True)


# TYPE 
# - descriptions are redundant
data.drop(["OFNS_DESC", "PD_DESC"], axis=1, inplace=True)
# - do not use PD_CD because there is too much unique values
data.drop("PD_CD", axis=1, inplace=True)


# SUCCEED
# - just use it and imputate missing values
data["CRM_ATPT_CPTD_CD"] = data["CRM_ATPT_CPTD_CD"].fillna(data["CRM_ATPT_CPTD_CD"].value_counts().index[0])


# DATETIME
# - ignore "to" date and time (a lot of missing values and no sense to use it)
data.drop(["CMPLNT_TO_DT", "CMPLNT_TO_TM"], axis=1, inplace=True)
# - extract and impute day of year, and day of week from CMPLNT_FR_DT
timestamp = pd.to_datetime(data["CMPLNT_FR_DT"], format="%m/%d/%Y", errors="coerce")
data["CMPLNT_DAY_OF_YEAR"] = timestamp.dt.day_of_year
data["CMPLNT_DAY_OF_YEAR"] = data["CMPLNT_DAY_OF_YEAR"].fillna(data["CMPLNT_DAY_OF_YEAR"].mean().round())
data["CMPLNT_DAY_OF_WEEK"] = timestamp.dt.day_of_week
data["CMPLNT_DAY_OF_WEEK"] = data["CMPLNT_DAY_OF_WEEK"].fillna(data["CMPLNT_DAY_OF_WEEK"].mean().round())
data.drop("CMPLNT_FR_DT", axis=1, inplace=True)
# - extract and impute second of day from CMPLNT_FR_TM
timestamp = pd.to_datetime(data["CMPLNT_FR_TM"], format="%H:%M:%S", errors="coerce")
data["CMPLNT_SECOND_OF_DAY"] = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second
data["CMPLNT_SECOND_OF_DAY"] = data["CMPLNT_SECOND_OF_DAY"].fillna(data["CMPLNT_SECOND_OF_DAY"].mean().round())
data.drop("CMPLNT_FR_TM", axis=1, inplace=True)

# RPT DATE
# - no sense to use it
data.drop("RPT_DT", axis=1, inplace=True)

# PREMISES
# - just use it
# - do not impute missing (unkown) values because xgboost can deal with NaN

# LOCALIZATION
# - no sense to use it
data.drop(["ADDR_PCT_CD", "BORO_NM", "JURIS_DESC", "JURISDICTION_CODE", "JURISDICTION_CODE", "PARKS_NM", "HADEVELOPT", "HOUSING_PSA", "X_COORD_CD", "Y_COORD_CD", "TRANSIT_DISTRICT", "Latitude", "Longitude", "Lat_Lon", "PATROL_BORO", "STATION_NAME"], axis=1, inplace=True)

# SUSP/VIC
# - do not impute missing values (unkown) and mark other/unkown as NaN for representation constistency
# - fix age groups
proper_age_group_pattern = r"<18|18-24|25-44|45-64|65+"
data.loc[~data["SUSP_AGE_GROUP"].str.match(proper_age_group_pattern, na=False), "SUSP_AGE_GROUP"] = np.nan
data.loc[~data["VIC_AGE_GROUP"].str.match(proper_age_group_pattern, na=False), "VIC_AGE_GROUP"] = np.nan
# - fix race
data.loc[(data["SUSP_RACE"] == "UNKNOWN") | (data["SUSP_RACE"] == "OTHER"), "SUSP_RACE"] = np.nan
data.loc[(data["VIC_RACE"] == "UNKNOWN") | (data["VIC_RACE"] == "OTHER"), "VIC_RACE"] = np.nan
# - fix sex
data.loc[(data["SUSP_SEX"] != "F") & (data["SUSP_SEX"] != "M"), "SUSP_SEX"] = np.nan
data.loc[(data["VIC_SEX"] != "F") & (data["VIC_SEX"] != "M"), "VIC_SEX"] = np.nan


data.to_csv("./NYPD_Complaint_Data_Historic_preprocessed.csv")
