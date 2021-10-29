import os
from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd


class IdentifierLabels:
    ID = "CMPLNT_NUM"


class DateTimeEventLabels:
    EVENT_START_DATE = "CMPLNT_FR_DT"
    EVENT_START_TIME = "CMPLNT_FR_TM"
    EVENT_END_DATE = "CMPLNT_TO_DT"
    EVENT_END_TIME = "CMPLNT_TO_TM"


class DateTimeSubmissionLabels:
    SUBMISSION_TO_POLICE_DATE = "RPT_DT"


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


RESULTS_DIR = "saved_plots/"


class ChartType(Enum):
    BAR = "Bar"
    LINE = "Line"


def plot_charts(dataset: pd.DataFrame) -> None:
    create_directory(RESULTS_DIR)
    dataset = dataset[dataset["SUSP_AGE_GROUP"].notna()]
    suspect_age = dataset.groupby([SuspectLabels.SUSPECT_AGE_GROUP])[IdentifierLabels.ID].count()
    victim_age = dataset.groupby([VictimLabels.VICTIM_AGE_GROUP])[IdentifierLabels.ID].count()

    print("suspect_age: ", suspect_age)
    print("victim_age: ", victim_age)
    draw_histograms_borough(dataset)
    draw_histograms_suspect(dataset)
    draw_histograms_victim(dataset)
    draw_map(dataset)
    draw_pie_plots(dataset)


def draw_pie_plots(dataset: pd.DataFrame) -> None:
    print_data_to_race_plots(dataset)
    print_data_to_sex_plots(dataset)
    # WHITE on WHITE  232682
    # WHITE on BLACK  29744
    # BLACK on WHITE  114036
    # BLACK on BLACK  769255
    # razem=1145717
    labels_race = 'WHITE on WHITE', 'WHITE on BLACK', 'BLACK on WHITE', 'BLACK on BLACK'
    races = [232682 / 1145717, 29744 / 1145717, 114036 / 1145717, 769255 / 1145717]

    plt.pie(races, labels=labels_race, autopct='%1.1f%%',
            shadow=False, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title("Race on race")
    save_plot("Race", "pie chart")

    # FEMALE on FEMALE     438353
    # FEMALE on MALE       226069
    # MALE on FEMALE       1140061
    # MALE on MALE         783518
    # razem=2588001
    labels_race = 'FEMALE on FEMALE', 'FEMALE on MALE', 'MALE on FEMALE', 'MALE on MALE'
    races = [438353 / 2588001, 226069 / 2588001, 1140061 / 2588001, 783518 / 2588001]

    plt.pie(races, labels=labels_race, autopct='%1.1f%%',
            shadow=False, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title("Sex on Sex")
    save_plot("Sex", "pie chart")


def print_data_to_race_plots(dataset: pd.DataFrame) -> None:
    dataset = dataset.groupby([SuspectLabels.SUSPECT_RACE, VictimLabels.VICTIM_RACE])[
        IdentifierLabels.ID].count()
    print("Race")
    print(dataset)


def print_data_to_sex_plots(dataset: pd.DataFrame) -> None:
    dataset = dataset.groupby([SuspectLabels.SUSPECT_SEX, VictimLabels.VICTIM_SEX])[
        IdentifierLabels.ID].count()
    print("Sex")
    print(dataset)


def draw_map(dataset: pd.DataFrame) -> None:
    # dataset = dataset[EventLocationLabels.LONGITUDE, EventLocationLabels.LATITUDE]
    # dataset = dataset[dataset.notna()]
    dataset = dataset.dropna(subset=[EventLocationLabels.LONGITUDE, EventLocationLabels.LATITUDE])
    dataset = dataset.astype({EventLocationLabels.LONGITUDE: str, EventLocationLabels.LATITUDE: str})
    # x = dataset[EventLocationLabels.LONGITUDE]
    # y = dataset[EventLocationLabels.LATITUDE]
    plt.scatter(x=dataset[EventLocationLabels.LONGITUDE], y=dataset[EventLocationLabels.LATITUDE],
                alpha=0.1)
    plt.title("Crimes in NYC")
    save_plot("Crimes in NYC", "Crimes in NYC")


def draw_histograms_borough(dataset: pd.DataFrame) -> None:
    dataset = dataset[dataset[EventLocationLabels.BOROUGH_NAME].notna()]
    dataset = dataset.astype({EventLocationLabels.BOROUGH_NAME: str})
    x = dataset[EventLocationLabels.BOROUGH_NAME]
    plt.hist(x, bins=9)
    # plt.ylim([float("1e5"), float("1e7")])
    plt.xlabel("Borough_name")
    plt.ylabel("Number")
    plt.title("Crimes in boroughs histogram")
    save_plot("Borough_name", "Number")


def draw_histograms_suspect(dataset: pd.DataFrame) -> None:
    labels = [SuspectLabels.SUSPECT_AGE_GROUP, SuspectLabels.SUSPECT_RACE, SuspectLabels.SUSPECT_SEX]
    bins = {SuspectLabels.SUSPECT_AGE_GROUP: 50,
            SuspectLabels.SUSPECT_RACE: 15,
            SuspectLabels.SUSPECT_SEX: 5}
    rotations = {SuspectLabels.SUSPECT_AGE_GROUP: 0,
                 SuspectLabels.SUSPECT_RACE: 60,
                 SuspectLabels.SUSPECT_SEX: 0}

    draw_histograms(dataset, labels, bins, rotations)


def draw_histograms_victim(dataset: pd.DataFrame) -> None:
    labels = [VictimLabels.VICTIM_AGE_GROUP, VictimLabels.VICTIM_RACE, VictimLabels.VICTIM_SEX]
    bins = {VictimLabels.VICTIM_AGE_GROUP: 120,
            VictimLabels.VICTIM_RACE: 15,
            VictimLabels.VICTIM_SEX: 9}
    rotations = {VictimLabels.VICTIM_AGE_GROUP: 0,
                 VictimLabels.VICTIM_RACE: 60,
                 VictimLabels.VICTIM_SEX: 0}

    draw_histograms(dataset, labels, bins, rotations)


def draw_histograms(dataset: pd.DataFrame, labels: [str], bins: {str: int},
                    rotation: {str: int}) -> None:
    plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
    for label in labels:
        dataset = dataset[dataset[label].notna()]
        dataset = dataset.astype({label: str})
        x = dataset[label]
        plt.hist(x, bins=bins[label])
        # plt.ylim([float("1e5"), float("1e7")])
        plt.xticks(rotation=rotation[label])
        plt.xlabel(label)
        plt.ylabel("Number")
        plt.title(label + " histogram")
        save_plot(label, "Number")


def cm_to_inch(value):
    return value / 2.54


def save_plot(x_axis_col_name: str, y_axis_col_name: str) -> None:
    plt.savefig(RESULTS_DIR + prepare_filename(f"{x_axis_col_name}#{y_axis_col_name}"),
                bbox_inches="tight")
    plt.close()
    plt.show()


def draw_plot(dataset: pd.DataFrame, x_axis_col_name: str,
              y_axis_col_name: str, chart_type: ChartType, save_data: bool) -> None:
    if chart_type == ChartType.LINE:
        plt.plot(dataset[x_axis_col_name], dataset[y_axis_col_name])
    elif chart_type == ChartType.BAR:
        plt.bar(dataset[x_axis_col_name], dataset[y_axis_col_name])

    plt.title(f"{x_axis_col_name} to {y_axis_col_name}")
    plt.xlabel(x_axis_col_name)
    plt.ylabel(y_axis_col_name)

    if save_data:
        plt.savefig(RESULTS_DIR + prepare_filename(f"{x_axis_col_name}#{y_axis_col_name}"))
        plt.close()
    plt.show()


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_filename(name: str, extension: str = "", add_date: bool = True) -> str:
    return (name + ("-" + datetime.now().strftime("%H%M%S") if add_date else "")
            + extension).replace(" ", "")


if __name__ == '__main__':
    df = pd.read_csv("../data/NYPD_Complaint_Data_Historic.csv")
    plot_charts(df)
