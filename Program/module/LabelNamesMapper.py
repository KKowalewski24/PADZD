from typing import List


class GeneralDataLabels:
    ID = "CMPLNT_NUM"


class DateTimeLabels:
    EVENT_START_DATE = "CMPLNT_FR_DT"
    EVENT_START_TIME = "CMPLNT_FR_TM"
    EVENT_END_DATE = "CMPLNT_TO_DT"
    EVENT_END_TIME = "CMPLNT_TO_TM"
    SUBMISSION_TO_POLICE_DATE = "RPT_DT"


class LawBreakingLabels:
    KEY_CODE = "KY_CD"
    OFFENSE_DESCRIPTION = "OFNS_DESC"
    PD_CODE = "PD_CD"
    PD_DESCRIPTION = "PD_DESC"
    LAW_BREAKING_STATUS = "CRM_ATPT_CPTD_CD"
    LAW_BREAKING_LEVEL = "LAW_CAT_CD"


class LocationLabels:
    PRECINCT_CODE = "ADDR_PCT_CD"
    BOROUGH_NAME = "BORO_NM"
    PLACE_TYPE_POSITION = "LOC_OF_OCCUR_DESC"
    PLACE_TYPE = "PREM_TYP_DESC"
    JURISDICTION_DESCRIPTION = "JURIS_DESC"
    JURISDICTION_CODE = "JURISDICTION_CODE"
    PARK_NAME = "PARKS_NM"
    NYC_HOUSING_DEVELOPMENT = "HADEVELOPT"
    DEVELOPMENT_LEVEl_CODE = "HOUSING_PSA"
    NYC_X_COORDINATE = "X_COORD_CD"
    NYC_Y_COORDINATE = "Y_COORD_CD"
    TRANSIT_DISTRICT_CODE = "TRANSIT_DISTRICT"
    LATITUDE = "Latitude"
    LONGITUDE = "Longitude"
    LATITUDE_LONGITUDE = "Lat_Lon"
    PATROL_DISTRICT_NAME = "PATROL_BORO"
    TRANSIT_STATION_NAME = "STATION_NAME"


class VictimSuspectLabels:
    SUSPECT_AGE_GROUP = "SUSP_AGE_GROUP"
    SUSPECT_RACE = "SUSP_RACE"
    SUSPECT_SEX = "SUSP_SEX"
    VICTIM_AGE_GROUP = "VIC_AGE_GROUP"
    VICTIM_RACE = "VIC_RACE"
    VICTIM_SEX = "VIC_SEX"


class LabelNamesMapper:
    general_data: GeneralDataLabels = GeneralDataLabels
    date_time: DateTimeLabels = DateTimeLabels
    law_breaking: LawBreakingLabels = LawBreakingLabels
    location: LocationLabels = LocationLabels
    victim_suspect: VictimSuspectLabels = VictimSuspectLabels


    @staticmethod
    def get_non_numeric_column_names() -> List[str]:
        # TODO ADD ALL FIELDS THAT ARE NOT NUMERIC
        return [
            LabelNamesMapper.law_breaking.LAW_BREAKING_STATUS,
            LabelNamesMapper.location.PATROL_DISTRICT_NAME,
            LabelNamesMapper.victim_suspect.SUSPECT_AGE_GROUP,
            LabelNamesMapper.victim_suspect.SUSPECT_RACE,
            LabelNamesMapper.victim_suspect.SUSPECT_SEX,
            LabelNamesMapper.victim_suspect.VICTIM_AGE_GROUP,
            LabelNamesMapper.victim_suspect.VICTIM_RACE,
            LabelNamesMapper.victim_suspect.VICTIM_SEX,
        ]
