from typing import List


class IdentifierLabels:
    ID = "CMPLNT_NUM"


class DateTimeEventLabels:
    EVENT_START_DATE = "CMPLNT_FR_DT"
    EVENT_START_TIME = "CMPLNT_FR_TM"
    EVENT_START_TIMESTAMP = "CMPLNT_FR_TIMESTAMP"
    EVENT_END_DATE = "CMPLNT_TO_DT"
    EVENT_END_TIME = "CMPLNT_TO_TM"
    EVENT_END_TIMESTAMP = "CMPLNT_TO_TIMESTAMP"


class DateTimeSubmissionLabels:
    SUBMISSION_TO_POLICE_DATE = "RPT_DT"
    SUBMISSION_TO_POLICE_TIMESTAMP = "RPT_TIMESTAMP"


class LawBreakingLabels:
    KEY_CODE = "KY_CD"
    OFFENSE_DESCRIPTION = "OFNS_DESC"
    PD_CODE = "PD_CD"
    PD_DESCRIPTION = "PD_DESC"
    LAW_BREAKING_LEVEL = "LAW_CAT_CD"


class EventStatusLabels:
    EVENT_STATUS = "CRM_ATPT_CPTD_CD"


class EventSurroundingsLabels:
    PLACE_TYPE = "PREM_TYP_DESC"
    PLACE_TYPE_POSITION = "LOC_OF_OCCUR_DESC"


class EventLocationLabels:
    PRECINCT_CODE = "ADDR_PCT_CD"
    BOROUGH_NAME = "BORO_NM"
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


class SuspectLabels:
    SUSPECT_AGE_GROUP = "SUSP_AGE_GROUP"
    SUSPECT_RACE = "SUSP_RACE"
    SUSPECT_SEX = "SUSP_SEX"


class VictimLabels:
    VICTIM_AGE_GROUP = "VIC_AGE_GROUP"
    VICTIM_RACE = "VIC_RACE"
    VICTIM_SEX = "VIC_SEX"


class LabelNamesMapper:
    identifier: IdentifierLabels = IdentifierLabels
    date_time_event: DateTimeEventLabels = DateTimeEventLabels
    date_time_submission: DateTimeSubmissionLabels = DateTimeSubmissionLabels
    law_breaking: LawBreakingLabels = LawBreakingLabels
    event_status: EventStatusLabels = EventStatusLabels
    event_surroundings: EventSurroundingsLabels = EventSurroundingsLabels
    event_location: EventLocationLabels = EventLocationLabels
    suspect: SuspectLabels = SuspectLabels
    victim: VictimLabels = VictimLabels


    @staticmethod
    def get_non_numeric_column_names() -> List[str]:
        # TODO ADD ALL FIELDS THAT ARE NOT NUMERIC
        return [

        ]
