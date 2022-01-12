class IdentifierLabels:
    ID = "CMPLNT_NUM"


class DateTimeEventLabels:
    EVENT_START_TIMESTAMP = "CMPLNT_FR_DT"
    EVENT_END_TIMESTAMP = "CMPLNT_TO_DT"


class DateTimeSubmissionLabels:
    SUBMISSION_TO_POLICE_TIMESTAMP = "RPT_DT"


class LawBreakingLabels:
    KEY_CODE = "KY_CD"
    PD_CODE = "PD_CD"
    LAW_BREAKING_LEVEL = "LAW_CAT_CD"


class EventStatusLabels:
    EVENT_STATUS = "CRM_ATPT_CPTD_CD"


class EventSurroundingsLabels:
    PLACE_TYPE = "PREM_TYP_DESC"
    PLACE_TYPE_POSITION = "LOC_OF_OCCUR_DESC"


class EventLocationLabels:
    PRECINCT_CODE = "ADDR_PCT_CD"
    BOROUGH_NAME = "BORO_NM"
    LATITUDE = "Latitude"
    LONGITUDE = "Longitude"


class SuspectLabels:
    SUSPECT_AGE_GROUP = "SUSP_AGE_GROUP"
    SUSPECT_RACE = "SUSP_RACE"
    SUSPECT_SEX = "SUSP_SEX"


class VictimLabels:
    VICTIM_AGE_GROUP = "VIC_AGE_GROUP"
    VICTIM_RACE = "VIC_RACE"
    VICTIM_SEX = "VIC_SEX"
