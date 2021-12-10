import os
import numpy as np
import pandas as pd


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
