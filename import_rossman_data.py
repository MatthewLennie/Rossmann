from torch.utils.data.dataset import Dataset
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
from typing import List

# Keys of the different types of variables.
cat_vars = [
    "Store",
    "DayOfWeek",
    "StateHoliday",
    "CompetitionMonthsOpen",
    "Promo2Weeks",
    "StoreType",
    "Assortment",
    "State",
    "Week",
    "Events",
    "Is_quarter_end_DE",
    "Is_quarter_start",
    "WindDirDegrees",
    "Is_quarter_start_DE",
    "Is_month_end",
    "Open",
    "Is_year_end",
    "Is_year_start_DE",
    "Is_month_start_DE",
    "Promo2",
    "Is_year_end_DE",
    "Dayofweek",
    "Is_month_start",
]

cont_vars = [
    "Sales",
    "Promo2SinceWeek",
    "Max_TemperatureC",
    "Mean_TemperatureC",
    "Min_TemperatureC",
    "Max_Humidity",
    "Mean_Humidity",
    "Min_Humidity",
    "Max_Wind_SpeedKm_h",
    "Mean_Wind_SpeedKm_h",
    "CloudCover",
    "trend",
    "trend_DE",
    "Promo",
    "SchoolHoliday",
    "Min_VisibilitykM",
    "Min_DewpointC",
    "Mean_VisibilityKm",
    "Precipitationmm",
    "MeanDew_PointC",
    "Mean_Sea_Level_PressurehPa",
    "Max_Sea_Level_PressurehPa",
    "Promo2Days",
    "Customers",
    "CompetitionDaysOpen",
    "Dew_PointC",
    "Dayofyear",
    "Min_Sea_Level_PressurehPa",
    "Max_Gust_SpeedKm_h",
    "Elapsed",
    "Max_VisibilityKm",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceYear",
]

weather_vars = [
    "Max_TemperatureC",
    "Mean_TemperatureC",
    "Min_TemperatureC",
    "Dew_PointC",
    "MeanDew_PointC",
    "Min_DewpointC",
    "Max_Humidity",
    "Mean_Humidity",
    "Min_Humidity",
    "Max_Sea_Level_PressurehPa",
    "Mean_Sea_Level_PressurehPa",
    "Min_Sea_Level_PressurehPa",
    "Max_VisibilityKm",
    "Mean_VisibilityKm",
    "Min_VisibilitykM",
    "Max_Wind_SpeedKm_h",
    "Mean_Wind_SpeedKm_h",
    "Max_Gust_SpeedKm_h",
    "Precipitationmm",
    "CloudCover",
    "WindDirDegrees",
]

output_file_name = "./data/joined_cleaned.pkl"


def data_clean(joined: pd.DataFrame) -> pd.DataFrame:
    """[function currently does basic na forward
    filling and conversion of variables to useful types.
    I also drop a bunch of columns that either are entirely null or
    duplciate columns, the data source seems to be a weirdly processed]

    Arguments:
        joined {df} -- [original df from kaggle download
        https://www.kaggle.com/init27/fastai-v3-rossman-data-clean]

    Returns:
        [df] -- [cleaned df]
    """
    joined.loc[:, weather_vars] = joined.loc[:, weather_vars].fillna(
        method="ffill"
    )

    weather_vars.append("Events")

    # some of the initial Max_Gust_Speed Data was missing
    # so I filled with the Max_wind Speed.
    joined.loc[
        joined["Max_Gust_SpeedKm_h"].isna(), "Max_Gust_SpeedKm_h"
    ] = joined.loc[joined["Max_Gust_SpeedKm_h"].isna(), "Max_Wind_SpeedKm_h"]

    #  change text data into categories, as codes.
    joined["Events"] = joined["Events"].astype("category").cat.codes + 1
    joined["Store"] = joined["Store"] - 1
    joined["DayOfWeek"] = joined["DayOfWeek"] - 1
    joined["Week"] = joined["Week"] - 1
    joined["Assortment"] = joined["Assortment"].astype("category").cat.codes
    joined["State"] = joined["State"].astype("category").cat.codes
    joined["WindDirDegrees"] = (
        joined["WindDirDegrees"].astype("category").cat.codes
    )
    joined["StoreType"] = joined["StoreType"].astype("category").cat.codes

    # Drop variables that didn't look useful.
    joined.drop(
        [
            "Promo2Since",
            "Year",
            "Month",
            "Day",
            "PromoInterval",
            "StateName",
            "file_DE",
            "State_DE",
            "Dayofweek_DE",
            "Day_DE",
            "Date",
            "Is_quarter_end",
            "Is_month_end_DE",
            "Is_year_start",
            "week",
            "file",
            "Month_DE",
            "week_DE",
            "Dayofyear_DE",
            "CompetitionOpenSince",
            "Date_DE",
            "Elapsed_DE",
            "CompetitionDistance",
        ],
        axis=1,
        inplace=True,
    )
    if "Id" in joined.keys():
        joined.drop("Id", axis=1, inplace=True)

    # check the keys. Make sure that we don't have a miss match
    # between keys in list and dataframe.
    a = set(joined.keys())
    total_keys = cat_vars.copy()
    total_keys.extend(cont_vars)
    b = set(total_keys)
    c = a.difference(b)
    assert not c

    # convert booleans to ints.
    joined[joined.select_dtypes(include="bool").keys()] = joined.select_dtypes(
        include="bool"
    ).astype("int")

    # change to floats.
    joined[cont_vars] = joined[cont_vars].astype("float")
    joined.dropna(0, inplace=True)
    return joined


class RossmanDataset(Dataset):
    """[puts data into a useful format to be used by the dataloader]
    """

    @classmethod
    def from_pickle(cls, pickle_file: str):
        """[creates the object from pickled dict, use to load pre-processed data]
        Arguments:
            pickle_file {[str]} -- [file name of pickled Rossmann Dataset.]
        """
        with open(pickle_file, "rb") as input:
            file = pickle.load(input)
        return file

    def to_pickle(self, output_file: str):
        """[puts the object into a pickle file for later recovery]

        Arguments:
            output_file {[str]} -- [output filename]
        """
        with open(output_file, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def __init__(
        self,
        df: pd.DataFrame,
        cont_vars: List[str],
        cat_vars: List[str],
        indices: List[int],
        scaler=MinMaxScaler(),
    ):

        # reading data, transforms etc..
        # column lists
        self.x_cols = df.columns.difference(["Sales", "Customers"])
        self.Y_cols = ["Sales", "Customers"]

        # scaler = MinMaxScaler()
        self.scaler = scaler

        # if statement on whether scaler has been set or not.
        if self.scaler == self.__init__.__defaults__[0]:

            # training case
            self.data = df.loc[indices, :].copy()

            # fit!!! and transform the continuous variables.
            self.data.loc[
                :, cont_vars + self.Y_cols
            ] = self.scaler.fit_transform(
                self.data.loc[:, cont_vars + self.Y_cols]
            )

        else:

            # validation case
            self.data = df.loc[indices, :].copy()

            # transform the continuous variables.
            self.data.loc[:, cont_vars + self.Y_cols] = self.scaler.transform(
                self.data.loc[:, cont_vars]
            )

        self.data.reset_index(inplace=True)
        self.data.drop(["index"], inplace=True, axis=1)

        # Make sure that the columsn have correct types
        self.x_data_cat = torch.tensor(
            self.data[cat_vars].values, dtype=torch.int
        )
        self.x_data_cont = torch.tensor(
            self.data[cont_vars].values, dtype=torch.float32
        )
        self.Y_data = torch.tensor(
            self.data[self.Y_cols].values, dtype=torch.float32
        )
        self.length = self.data.shape[0]

    def __getitem__(self, index):
        # returns the input and output
        return (
            self.x_data_cat[index],
            self.x_data_cont[index],
            self.Y_data[index],
        )

    def __len__(self):
        return self.length  # of how many examples(images?) you have


if __name__ == "__main__":

    # Example usage
    # just used the joined dataframes
    joined = pd.read_pickle("./data/joined")

    # joined_test doesn't contain customers or sales.
    # they are the predicted variables.
    joined_test = pd.read_pickle("./data/joined_test")

    # push through data clean function
    # i.e. drop nonesense columns and fill nans
    joined = data_clean(joined)

    # train valid splitting
    split_train = int(joined.shape[0] * 0.8)
    split_valid = joined.shape[0] - split_train
    train, valid = torch.utils.data.random_split(
        joined, [split_train, split_valid]
    )

    # create and save the training set
    train_data = RossmanDataset(joined, cont_vars, cat_vars, train.indices)
    train_data.to_pickle("./data/train_data.pkl")

    # create and save the validation set using the scaler
    # set in the training set.
    valid_data = RossmanDataset(
        joined, cont_vars, cat_vars, valid.indices, scaler=train_data.scaler
    )
    valid_data.to_pickle("./data/valid_data.pkl")
