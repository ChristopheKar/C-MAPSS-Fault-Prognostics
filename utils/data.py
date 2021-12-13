import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold


def load_cmapss_subset(data_dir, subset, unit_col='unit'):
    """
    Load one of 4 subsets of NASA C-MAPSS dataset.

    Parameters
    ----------
    data_root : str
                Path to directory containing dataset files.
    subset: int
            Data subset to load, generally between 1 and 4.

    Returns
    -------
    train: numpy.ndarray
        Array containing subset training data.
    test: numpy.ndarray
        Array containing subset testing data.
    header: list of strings
        List containing data column names
    n_meta: int
        Number of non-sensor columns
    n_sensors: int
        Number of sensor columns
    n_settings: int
        Number of setting columns
    """

    subset = 'FD' + str(subset).zfill(3)
    subset_path = os.path.join(data_dir, '{}_' + subset + '.txt')

    n_settings = 3
    n_sensors = 21
    n_meta = 2 + n_settings

    header = [unit_col, 'time']
    header += [f'setting{i+1}' for i in range(n_settings)]
    header += [f'sensor{i+1}' for i in range(n_sensors)]

    train = pd.read_csv(
        subset_path.format('train'),
        sep='\s+', header=None, names=header)
    train = add_rul(train, unit_col)
    test = pd.read_csv(
        subset_path.format('test'),
        sep='\s+', header=None, names=header)
    test = add_rul(test, unit_col)
    y_test = np.loadtxt(subset_path.format('RUL'))
    test['RUL'] += np.repeat(y_test, test.groupby(by=unit_col)['time'].max())

    return train, test, header, n_meta, n_sensors, n_settings


def add_rul(df, unit_col='unit'):
    """Add remaining useful life to dataframe rows."""
    # Group dataframe by unit and get largest-time sample
    max_time = df.groupby(by=unit_col)['time'].max()

    # Merge the max cycle back into the original frame
    merged_df = df.merge(
        max_time.to_frame(name='max_time'),
        left_on=unit_col,
        right_index=True)

    # For each row (current time), RUL = (max time) - (current time)
    merged_df['RUL'] = merged_df['max_time'] - merged_df['time']
    merged_df = merged_df.drop('max_time', axis=1)

    return merged_df


def add_lag(df_in, n_lags, columns):
    """Add lagged variables to dataframe."""
    df = df_in.copy()
    for i in range(n_lags):
        lagged_columns = [col + '_lag_{}'.format(i+1) for col in columns]
        df[lagged_columns] = df.groupby('unit')[columns].shift(i+1)
    df.dropna(inplace=True)
    return df


def sorter(name):
    """Sort by sensor index."""
    if ('sensor' in name):
        return int(name.split('sensor')[-1].split('_')[0])
    else:
        return 0


def build_dataset(
    df,
    drop_sensors=None,
    clip=None, cv_folds=5,
    smooth=0, lag=0,
    test=False, include_settings=None,
    return_cols=False, reshape_2d=False):

    """
    Process dataset for training and testing.

    Parameters
    ----------
    df: pd.DataFrame
        C-MAPSS train or test data.
    drop_sensors: list or None
        List of sensor indexes to drop.
    clip: int or None
        Saturate y-train values at `clip` value.
    cv_folds: int or None
        Number of folds for cross-validation.
    smooth: int
        Window size for moving-average smoother, 0 for no smoothing.
    lag: int
        Number of lagged variables to add, 0 for none.
    test: bool
        Specifies if dataset is test or train.
    include_settings: list or None
        List of setting indexes to include.
    return_cols: bool
        If True, returns column names.
    reshape_2d: bool
        If True and `lag` > 0, reshapes data from (samples, n_sensors x lag),
        to (samples, n_sensors, lag).

    """

    header = list(df.columns)[:-1]
    n_meta = len(header) - len([col for col in header if ('sensor' in col)])

    if (drop_sensors is not None):
        drop_sensors = [f'sensor{i}' for i in drop_sensors]
        df = df.drop(drop_sensors, axis=1)

    if (lag > 0):
        df = add_lag(df, lag, df.columns[n_meta:-1])


    if (test):
        smooth = 0
        clip = None
        df = df.groupby(['unit']).last().reset_index()

    if (include_settings is not None):
        columns = list(set(header[:n_meta]) - set([f'setting{s}' for s in include_settings]))
    else:
        columns = header[:n_meta]


    X = df.drop(columns, axis=1)
    y = X.pop('RUL')
    cols = sorted(X.columns, key=sorter)
    X = X[cols]

    if (smooth > 0):
        X = X.rolling(smooth, min_periods=1).mean()

    X = np.asarray(X)
    y = np.asarray(y)

    if ((reshape_2d) and (lag > 0)):
        X = X.reshape((-1, len(cols)//(lag + 1), lag + 1))

    if (isinstance(clip, int)):
        y = y.clip(max=clip)


    if (cv_folds is not None):
        try:
            cv_folds = int(cv_folds)
        except:
            raise TypeError('`cv_folds` must be None or int')

        cv = KFold(n_splits=cv_folds)
    else:
        cv = None

    if (return_cols):
        return X, y, cv, cols
    else:
        return X, y, cv
