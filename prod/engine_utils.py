from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from pandas import Series
from toolz.curried import filter, map, pipe

DEAD_COHORT_DX_DELTA_DAYS = 30


def df_range(df, start_date, end_date):
    return df[(df.index >= start_date) & (df.index <= end_date)]


def strip_excess_dates(df, start_date, end_date):
    df_stripped = df.copy()

    df_stripped = df_stripped[
        (df_stripped.index >= start_date) & (df_stripped.index <= end_date)
    ]

    df_stripped = df_stripped.reindex([i for i in range(0, len(df_stripped))], axis=1)
    return df_stripped


def round_user(x):
    # Helper method to round user up if >= 0.5 and down if < 0.5
    x = np.array(x)
    x = x + 0.0000000001
    return np.round(x)


def round_user_df(x):
    # Helper method to round user up if >= 0.5 and down if < 0.5
    x = x + 0.0000000001
    return np.round(x)


# Helper function to calculate primary and secondary diagonal of a square matrix
def diagonal_sums_v2(mat, n):
    return np.trace(mat, offset=n)


def flatten_pivot(df, dataframe=True):
    dt = df.copy()
    dt = dt.fillna(0)

    mat = np.rot90(dt.values)
    sums = [diagonal_sums_v2(mat, i) for i in range(-len(mat) + 1, 1)]

    if dataframe:
        return pd.DataFrame(sums, index=dt.index)
    else:
        return sums


def absolute_seasonality(x):
    return np.sum(np.abs(x)) / len(x)


def gen_mask(n):
    """
    gen_mask takes in n number of dimensions and generates boolean mask
    This mask corresponds to the first diagonal and all values above that, rotated 90 degrees anticlockwise
    """
    a = np.ones((n, n), bool)
    np.fill_diagonal(a, False)

    for i in range(0, n):
        x = np.arange(i)
        a[x, i] = False

    return np.rot90(a)


def gen_mask_upper_triangle(dimension, mask_from_row):
    a = np.ones((dimension, dimension), bool)

    for i in range(0, mask_from_row):
        x = np.arange(mask_from_row - i)
        a[x, i] = False

    return a


def gen_mask_lower_triangle(dimension, mask_from_row):
    a = np.zeros((dimension, dimension), bool)

    for i in range(0, mask_from_row):
        x = np.arange(mask_from_row - i)
        a[x, i] = True

    return a


def to_dim_date(date):
    return int(date.strftime("%Y%m%d"))


def to_date(dim_date):
    return datetime.strptime(str(dim_date), "%Y%m%d")


def strip_outliers(yactual, z_score=1.96):
    """
    This method takes in a single column dependant variable, such as retention,
    1) converts the array to a dataframe,
    2) computes the z-score of the dataframe
    3) filters  the input dataframe z-values on a selected confidence interval (z-score = 1.96 => 97.5%)
    4) returns  the yactual series with outliers beyond the confidence interval removed.
    """
    df_in = pd.DataFrame(yactual)
    df_score = df_in.transform(lambda col: (col - col.mean()) / col.std())
    yactual_out = df_in[df_score < z_score].values
    return yactual_out


def create_and_reindex_series(x_data, y_data, x_forecast):
    return Series(y_data, index=x_data).reindex(x_forecast)


def format_date_column(df, column):
    df[column] = pd.to_datetime(df[column])
    df[column] = df[column].dt.tz_localize(None)
    df[column] = df[column].dt.normalize()
    return df


def calculate_days_since_install(
    df: pd.DataFrame, date_col="cohort_date", activity_col="calendar_date"
) -> pd.DataFrame:
    return df.assign(dx=lambda df: (df[activity_col] - df[date_col]).dt.days)


def generate_ranged_clf_dataframe(start_date, end_date) -> pd.DataFrame:
    """
    Takes in start and end dates:
    start_date, end_date       datetime64[ns]/string

    Generates a daily cohort-long-form dataframe with columns:

    cohort_date      datetime64[ns]
    calendar_date    datetime64[ns]
    dx                        int64
    """
    input_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    index = pd.MultiIndex.from_product(
        [input_date_range, input_date_range], names=["cohort_date", "calendar_date"]
    )
    return (
        index[
            index.get_level_values("calendar_date")
            >= index.get_level_values("cohort_date")
        ]
        .to_frame(index=False)
        .pipe(calculate_days_since_install)
    )


def extract_clf_cohort_size(
    df,
    date_col="cohort_date",
    activity_col="calendar_date",
    query_col="daily_active_users",
) -> pd.DataFrame:
    """
    Using cohort long form dataframe, extract cohort size
    """
    cohort_size = df.query(f"{activity_col} == {date_col}")[
        [query_col, date_col]
    ].rename(columns={query_col: "cohort_size"})
    return df.merge(cohort_size, on=date_col)


def limit_clf_df_to_actual_input(df, config) -> pd.DataFrame:
    """
    Using engine config object limit clf dataframe to actual input range
    """
    return (
        df.query(
            f'cohort_date >= "{config.start_input_range}" and cohort_date <= "{config.end_input_range}"'
        )
        .query(
            f'calendar_date >= "{config.start_input_range}" and calendar_date <= "{config.end_input_range}"'
        )
        .query("cohort_date <= calendar_date")
    )


def reindex_and_trim_clf_input_data(df, config) -> pd.DataFrame:
    """
    Using engine config object reidex and trim cohort long form dataframe to actual input range
    """
    actual_range_multi_index = (
        generate_ranged_clf_dataframe(config.start_input_range, config.end_input_range)
        .set_index(["cohort_date", "calendar_date"])
        .sort_index()
        .index
    )
    return (
        df.set_index(["cohort_date", "calendar_date"])
        .reindex(actual_range_multi_index)
        .reset_index()
        .pipe(limit_clf_df_to_actual_input, config)
        .replace(np.nan, 0)
    )


def calculate_clf_df_shape(date_range: int):
    """
    Reindexing a clf dataframe generate a flattened triangle of shape:
    1/2 ( date_range + 1) * date_range

    This method generates and returns the size of that dataframe
    """
    return ((date_range + 1) / 2) * date_range


def drop_indices(df, query):
    idx = df.query(query).index
    return df.drop(idx)


def convert_wide_to_long(df, dx_col="dx", date_col="cohort_date", data_col="retention"):
    # Ensure columns and index not named, otherwise df won't unstack into columns with generic names level_0 and level_1
    if df.empty:
        return df

    df.columns.name = None
    df.index.name = None
    return (
        df.unstack()
        .reset_index()
        .rename(columns={"level_0": dx_col, "level_1": date_col, 0: data_col})
    )


def limit_clf_input_to_lookback_config(df, config):
    """
    Filters CLF input dataframe to both standard and temporal lookback ranges.
    Parameters
    ----------
    df: pandas.DataFrame
                            Index:                              (RangeIndex)
                            Columns:
                                    - cohort_date               (datetime64[ns])
                                    - calendar_date             (datetime64[ns])
                                    - dx                        (int)
    config: engine.config.Config
        model_start_date_range:                                 (datetime64[ns])
        end_input_range:                                        (datetime64[ns])

        apply_temporal_lookback:                                (bool)
        temporal_lookback                                       (int)
    """
    if config.apply_temporal_lookback:
        calendar_date_start = config.end_input_range - timedelta(
            config.temporal_lookback - 1
        )
        cohort_date_start = config.model_start_date_range - timedelta(
            config.temporal_lookback - 1
        )
    else:
        calendar_date_start = config.model_start_date_range
        cohort_date_start = config.model_start_date_range
    return (
        df.query(
            f"calendar_date <= '{config.end_input_range}' and calendar_date >= '{calendar_date_start}'"
        )
        .query(
            f"cohort_date <= '{config.end_input_range}' and cohort_date >= '{cohort_date_start}'"
        )
        .query(f"cohort_date >= '{config.start_input_range}'")
        .query(f"dx < {config.lookback}")
    )


def calculate_activity_date(
    df, dx_col="dx", date_col="cohort_date", output_col="calendar_date"
):
    if df.empty:
        return df

    timedelta_index = pd.TimedeltaIndex(df.loc[:, dx_col], unit="D")
    df.loc[:, output_col] = df.loc[:, date_col] + timedelta_index
    return df


def cast_and_select_column_subset(df, columns, dtypes):
    """
    Takes in a dataframe, creates a casting dictionary and returns only columns of interest.
    Parameters
    ----------
    df: pandas.DataFrame
    columns: List
    dtypes: List of datatypes
    """

    casting_dict = dict(zip(columns, dtypes))
    return df.astype(casting_dict)[columns]


def safe_concat(dataframes, columns, dtypes):
    """
    Takes in a dataframe, creates a casting dictionary and returns only columns of interest.
    Parameters
    ----------
    dataframes: List of pandas.DataFrames
    columns: List
    dtypes: List of datatypes
    """

    entries = pipe(
        dataframes,
        filter(lambda entry: entry is not None),
        filter(lambda df: df.shape[0] != 0),
        map(
            lambda df: cast_and_select_column_subset(df, columns=columns, dtypes=dtypes)
        ),
    )

    return pd.concat(entries)


def generate_dx_window_list(
    start_input_range: datetime,
    model_end_date_range: datetime,
    daily_resolution: bool,
) -> List[int]:
    """Returns list of dx points to be persisted"""
    dx_max = (model_end_date_range - start_input_range).days + 1
    if daily_resolution:
        return [dx for dx in range(0, dx_max)]

    sparse_windows = [0, 1, 7, 14]
    ten_day_windows = [i for i in range(10, dx_max + 1, 10)]
    annual_windows = [i for i in range(365, dx_max + 1, 365)]
    windows = sparse_windows + ten_day_windows + annual_windows
    dx_filter = [360, 1090]
    return sorted({dx for dx in windows if dx not in dx_filter and dx < dx_max})


def limit_clf_to_date_range(df, start_date, end_date):
    return df[
        (df["cohort_date"] >= start_date)
        & (df["cohort_date"] <= end_date)
        & (df["calendar_date"] <= end_date)
    ]


def weighted_mean(df, value_col, weight_col):
    """
    Calculate mean of value column weighted by other feature column.
    """
    if all(df[value_col].isna()):
        return np.nan
    if all(df[weight_col] == 0):
        return 0
    else:
        return np.nansum(df[value_col] * df[weight_col]) / np.nansum(df[weight_col])


def dx_window_median(df, numerator, denominator):
    """
    Grouping by days since install, calculate the medain over all days since install
    """
    median = (
        df[[numerator, denominator, "dx"]]
        .groupby("dx")
        .sum()
        .eval(f"{numerator}/{denominator}")
        .median()
    )
    return median


def dx_window_mean(df, numerator, denominator):
    """
    Calculate the mean over lookback window
    """
    return np.nansum(df[numerator]) / np.nansum(df[denominator])


def limit_extended_forecast(df, x_col, y_col, buyback_window, persist_model_end):
    """
    Trim data outside buyback window from extended forecast.

    :param pd.DataFrame df:
    :param string x_col:
    :param string y_col:
    :param int buyback_window:
    :param string persist_model_end:
    :return:
    :rtype: pd.DataFrame
    """
    df = df.loc[df[x_col] <= persist_model_end]

    df = df.assign(buyback_end_date=df[x_col] + timedelta(buyback_window))
    df = df.loc[
        (df[y_col] <= df["buyback_end_date"]) | (df[y_col] <= persist_model_end)
    ]
    return df.drop(columns=["buyback_end_date"])


def limit_extended_forecast_clv(df, buyback_window, persist_model_end):
    """
    Trim CLV data outside buyback window from extended forecast.

    :param pd.DataFrame df:
    :param int buyback_window:
    :param string persist_model_end:
    :return:
    :rtype: pd.DataFrame
    """
    df = df.loc[df.index <= persist_model_end]

    df = df.assign(persist_window=(persist_model_end - df.index).days)
    df = df.loc[
        (df.clv_window <= buyback_window) | (df.clv_window <= df.persist_window)
    ]
    return df.drop(columns=["persist_window"])


def extrapolate_override_vector(df, value_col, max_date):
    """
    Assigns last, non-null, value in override dataframe to all dates greater than max date.
    Parameters
    ----------
    df : pandas.DataFrame
        Index:
            Dtype: datetime64
        Columns:
            Name: value_col, Any
    value_col: string
    max_date : datetime.datetime
    """
    extrapolation_value = df.query(f'index == "{max_date}"')[value_col].values[0]

    df[value_col] = np.where(df.index > max_date, extrapolation_value, df[value_col])

    return df


def get_filtered_actual_input_df(
    input_df: pd.DataFrame,
    block_start_date: datetime,
    minimum_cohort_contribution_size: int,
    block_size: int,
) -> pd.DataFrame:
    """
    Returns the first block_size number of cohorts after block_start_date of at least size minimum_cohort_contribution_size
    Parameters
    ----------
    input_df : pandas.DataFrame
        Columns:
            Name: cohort_date, dtype: datetime64
            Name: dx, dtype: int64
            Name: cohort_size, dtype: object
    """
    valid_cohorts = (
        input_df.dropna()
        .query(f"cohort_date >= '{block_start_date}'")
        .query("dx == 0")
        .query(f"cohort_size >= {minimum_cohort_contribution_size}")
        .nsmallest(block_size, "cohort_date")["cohort_date"]
        .values
    )
    return input_df.query("cohort_date in @valid_cohorts")


def is_sufficient_no_of_cohorts(
    filtered_actual_input_df: pd.DataFrame,
    block_size: int,
) -> bool:
    """
    Determines whether at least block_size number of cohorts present
    Parameters
    ----------
    filtered_actual_input_df : pandas.DataFrame
        Columns:
            Name: dx, dtype: int64
    """
    if len(filtered_actual_input_df.query("dx == 0")) < block_size:
        return False
    return True


def get_block_cutoff_dx(df: pd.DataFrame) -> int:
    """
    Gets the maximum dx value of youngest cohort should this value be at least 31. Else, gets the
    max dx value of the oldest cohort.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns:
            Name: cohort_date, dtype: datetime
            Name: dx, dtype: int64
    """
    block_cutoff_dx = df.query("cohort_date == cohort_date.max()")["dx"].max()
    if block_cutoff_dx <= 30:
        block_cutoff_dx = df.query("cohort_date == cohort_date.min()")["dx"].max()
    return block_cutoff_dx


def generate_dead_cohort_multiplier(df: pd.DataFrame, end_input_range, column):
    return (
        df.pipe(calculate_activity_date)
        .query(
            f'calendar_date <= "{end_input_range}" and calendar_date >= "{end_input_range  - timedelta(days=DEAD_COHORT_DX_DELTA_DAYS)}"'
        )
        .pivot("calendar_date", "cohort_date", column)
        .median()
        .rename("cohort_multiplier")
        .apply(lambda col: col != 0)
        .astype(int)
    )
