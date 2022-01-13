import numpy as np
import pandas as pd


def basic_stats(series):
    print("Number of unique values:", len(series.unique()))
    print("Number of NA:", np.count_nonzero(series.isna()))
    series.value_counts().plot(kind='bar')


data = pd.read_csv("./NYPD_Complaint_Data_Historic.csv")

###########################################################################
# INDEX
basic_stats(data["CMPLNT_NUM"])
###########################################################################


###########################################################################
# TYPE

# basic stats
basic_stats(data["KY_CD"])
basic_stats(data["OFNS_DESC"])
basic_stats(data["PD_CD"])
basic_stats(data["PD_DESC"])
basic_stats(data["LAW_CAT_CD"])
# ky_cd and pd_cd are completely inbalanced

# associations between codes and descriptions
desc_per_key = data[["KY_CD", "OFNS_DESC"]].groupby("KY_CD")
for key, desc in desc_per_key:
    print("=====================================")
    print(key)
    print(desc["OFNS_DESC"].value_counts())
desc_per_pd = data[["PD_CD", "PD_DESC"]].groupby("PD_CD")
for pd, desc in desc_per_pd:
    print("=====================================")
    print(key)
    print(desc["PD_DESC"].value_counts())
# descriptions are almost one to one (only a few times two or three similar
# descriptions to the same code/pd) - descriptions are generally redundant

# association between KY_CD and PD_CD
ky_per_pd = data[["PD_CD", "KY_CD"]].groupby("PD_CD")
for pd, ky in ky_per_pd:
    print("=====================================")
    print(pd)
    print(ky["KY_CD"].value_counts())
# almost one to many, there are some outliers-like incosistencies, there is also a couple
# of pd_cd which matches two or more different ky_cd and don't look like errors - because of
# these couple examples, storing ky_cd with pd_cd should be informative and ky_cd
# seems not to be redundant

# association between LAW_CAT_CD and KY_CD
cat_per_key = data[["KY_CD", "LAW_CAT_CD"]].groupby("KY_CD")
for key, cat in cat_per_key:
    print("=====================================")
    print(key)
    print(cat["LAW_CAT_CD"].value_counts())
# it's also almost one to many, but there is a few keys, which contains numerous
# samples for two or three categories - because of these samples,
# storing law_cat_cd with ky_cd should be infomative
###########################################################################


###########################################################################
# SUCCEED
basic_stats(data["CRM_ATPT_CPTD_CD"])
# completely inbalanced!
###########################################################################


###########################################################################
# DATETIME
basic_stats(data["CMPLNT_FR_DT"])
basic_stats(data["CMPLNT_FR_TM"])
basic_stats(data["CMPLNT_TO_DT"])
basic_stats(data["CMPLNT_TO_TM"])
###########################################################################


###########################################################################
# RPT DATE
basic_stats(data["RPT_DT"])
###########################################################################


###########################################################################
# PREMISES
basic_stats(data["PREM_TYP_DESC"])
basic_stats(data["LOC_OF_OCCUR_DESC"])
###########################################################################


###########################################################################
# SUSP/VIC
basic_stats(data["SUSP_AGE_GROUP"])
basic_stats(data["VIC_AGE_GROUP"])
basic_stats(data["SUSP_RACE"])
basic_stats(data["VIC_RACE"])
basic_stats(data["SUSP_SEX"])
basic_stats(data["VIC_SEX"])

# after age group cleaning
proper_age_group_pattern = r"<18|18-24|25-44|45-64|65+"

susp_age_group = data["SUSP_AGE_GROUP"]
susp_age_group[~susp_age_group.str.match(proper_age_group_pattern, na=False)] = np.nan
basic_stats(susp_age_group)

vic_age_group = data["VIC_AGE_GROUP"]
vic_age_group[~vic_age_group.str.match(proper_age_group_pattern, na=False)] = np.nan
basic_stats(vic_age_group)
###########################################################################


###########################################################################
# LOCATION
basic_stats(data["ADDR_PCT_CD"])
basic_stats(data["BORO_NM"])
basic_stats(data["JURIS_DESC"])
basic_stats(data["JURISDICTION_CODE"])
basic_stats(data["PARKS_NM"])
basic_stats(data["HADEVELOPT"])
basic_stats(data["HOUSING_PSA"])
basic_stats(data["X_COORD_CD"])
basic_stats(data["Y_COORD_CD"])
basic_stats(data["TRANSIT_DISTRICT"])
basic_stats(data["Latitude"])
basic_stats(data["Longitude"])
basic_stats(data["Lat_Lon"])
basic_stats(data["PATROL_BORO"])
basic_stats(data["STATION_NAME"])
