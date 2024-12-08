import pandas as pd
def ingest_data(start_date, end_date):
    # Load and filter the train.csv data
    df_train = (
        pd.read_csv('train.csv', parse_dates=['date'])
        .query('date >= @start_date and date <= @end_date')
    )

    df_oil = (
        pd.read_csv('oil.csv', parse_dates=['date'])
        .query('date >= @start_date and date <= @end_date')
    )

    return df_train,df_oil
