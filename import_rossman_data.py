import pandas as pd
import sklearn.impute
import numpy as np
cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
            'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
            'State', 'Week', 'Events', 'Is_quarter_end_DE', 'Is_quarter_start', 'WindDirDegrees', 'Is_quarter_start_DE', 'Is_month_end',
            'Open', 'Is_year_end', 'Is_year_start_DE', 'Is_month_start_DE', 'Promo2', 'Is_year_end_DE', 'Dayofweek', 'Is_month_start', 'StateName']

cont_vars = ['Sales', 'Promo2SinceWeek', 'CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
             'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h',
             'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
             'Promo', 'SchoolHoliday', 'Min_VisibilitykM', 'Min_DewpointC', 'Mean_VisibilityKm', 'Precipitationmm', 'MeanDew_PointC',
             'Mean_Sea_Level_PressurehPa', 'Max_Sea_Level_PressurehPa', 'Promo2Days',
             'Customers', 'CompetitionDaysOpen', 'Dew_PointC', 'Dayofyear', 'Min_Sea_Level_PressurehPa', 'Max_Gust_SpeedKm_h', 'Elapsed', 'Max_VisibilityKm', 'CompetitionOpenSinceMonth']

weather_vars = ['Max_TemperatureC', 'Mean_TemperatureC',
                'Min_TemperatureC', 'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC',
                'Max_Humidity', 'Mean_Humidity', 'Min_Humidity',
                'Max_Sea_Level_PressurehPa', 'Mean_Sea_Level_PressurehPa',
                'Min_Sea_Level_PressurehPa', 'Max_VisibilityKm', 'Mean_VisibilityKm',
                'Min_VisibilitykM', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h',
                'Max_Gust_SpeedKm_h', 'Precipitationmm', 'CloudCover',
                'WindDirDegrees']

output_file_name = "joined_cleaned.pkl"
def data_clean(joined):
    """[function currently does basic na forward filling and conversion of variables to useful types. 
    I also drop a bunch of columns that either are entirely null or duplciate columns]

    Arguments:
        joined {df} -- [original df from kaggle download https://www.kaggle.com/init27/fastai-v3-rossman-data-clean]

    Returns:
        [df] -- [cleaned df]
    """
    joined.loc[:, weather_vars] = joined.loc[:,
                                             weather_vars].fillna(method='ffill')
    weather_vars.append('Events')
    # some of the initial Max_Gust_Speed Data was missing so I filled with teh Max_wind Speed. 
    joined.loc[joined['Max_Gust_SpeedKm_h'].isna(
    ), 'Max_Gust_SpeedKm_h'] = joined.loc[joined['Max_Gust_SpeedKm_h'].isna(), 'Max_Wind_SpeedKm_h']
    #  change text data into categories, as codes.
    joined['Events'] = joined['Events'].astype('category').cat.codes + 1
    joined['Assortment'] = joined['Assortment'].astype('category').cat.codes
    joined['State'] = joined['State'].astype('category').cat.codes
    joined['StoreType'] = joined['StoreType'].astype('category').cat.codes
    joined.drop(['Promo2Since', 'PromoInterval', 'StateName', 'file_DE', 'State_DE', 'Dayofweek_DE', 'Day_DE', 'Date', 'Is_quarter_end', 'Is_month_end_DE',
                 'Is_year_start', 'week', 'file', 'Month_DE', 'week_DE', 'Dayofyear_DE', 'CompetitionOpenSince', 'Date_DE', 'Elapsed_DE'], axis=1, inplace=True)
    if 'Id' in joined.keys():
        joined.drop('Id', axis=1, inplace=True)

    # check the keys. Make sure that we don't have a miss match between keys in list and dataframe.
    a = set(joined.keys())
    total_keys = cat_vars.copy()
    total_keys.extend(cont_vars)
    b = set(total_keys)
    c = a.difference(b)
    assert not c
    # convert booleans to ints.
    joined[joined.select_dtypes(include='bool').keys()] = joined.select_dtypes(
        include='bool').astype('int')
    # change to floats.
    joined[cont_vars] = joined[cont_vars].astype('float')
    return joined


if __name__ == "main":
    # just used the joined dataframes
    joined = pd.read_pickle('./data/joined')
    # joined_test doesn't contain customers or sales. they are the predicted variables. 
    joined_test = pd.read_pickle('./data/joined_test')
    joined = data_clean(joined)
    joined.to_pickle(output_file_name)